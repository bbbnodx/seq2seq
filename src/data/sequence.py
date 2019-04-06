# coding: utf-8
from common.np import *
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import chain

class TextSequence:
    '''
    テキストシーケンスのクラス
    1行のテキストを文字単位に分割し、文字IDのベクトルに変換する
    また、CSVの入出力メソッドを持つ
    '''
    def __init__(self, SOS='<SOS>', EOS='<EOS>', PAD='<PAD>', UNK='<UNK>'):
        '''
        Parameters
        ----------
        SOS : str, optional
            教師データの文章の開始を示す特殊文字
        EOS : str, optional
            文章の終わりを示す特殊文字
            教師データのみに用いられる
        PAD : str, optional
            文章の長さを均一にするために挿入される特殊文字
        UNK : str, optional
            辞書に存在しないトークンが用いられたとき、代わりに挿入される特殊文字
        '''

        self.__SOS = SOS
        self.__EOS = EOS
        self.__PAD = PAD
        self.__UNK = UNK
        self._init_vocab()
        self.vec_x = None
        self.vec_t = None

    def _init_vocab(self):
        self.char_to_id = {self.__PAD: 0,
                           self.__SOS: 1,
                           self.__EOS: 2,
                           self.__UNK: 3}
        self.id_to_char = {0: self.__PAD,
                           1: self.__SOS,
                           2: self.__EOS,
                           3: self.__UNK}

    @property
    def SOS(self):
        return self.__SOS

    @property
    def EOS(self):
        return self.__EOS

    @property
    def PAD(self):
        return self.__PAD

    @property
    def PAD_id(self):
        return self.char_to_id[self.__PAD]

    @property
    def UNK(self):
        return self.__UNK

    @property
    def start_id(self):
        return self.char_to_id[self.__SOS]

    @property
    def sample_size(self):
        return self.vec_t.shape[1] - 1  # exclude SOS

    @property
    def vocab(self):
        return self.char_to_id, self.id_to_char

    @property
    def vocab_size(self):
        return len(self.char_to_id)

    @vocab.setter
    def vocab(self, vocab):
        '''
        Parameters
        ----------
        vocab : tuple(dict, dict)
            char_to_id, id_to_char
        '''
        char_to_id, id_to_char = vocab
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char

    @property
    def raw_x(self):
        '''
        文字列はデータとして持たず、ベクトル表現から変換して返す
        '''
        return self.vec2seq(self.vec_x)

    @property
    def raw_t(self):
        '''
        文字列はデータとして持たず、ベクトル表現から変換して返す
        '''
        return self.vec2seq(self.vec_t)

    def _update_vocab(self, text):
        '''
        ボキャブラリ辞書の更新

        Parameters
        ----------
        text : str
            読み込む文字列
        '''
        for char in text:
            if char not in self.char_to_id:
                i = len(self.char_to_id)
                self.char_to_id[char] = i
                self.id_to_char[i] = char

    def read_csv(self, source_csv, col_x='x', col_t='y', vocab_init=False, vocab_update=True):
        '''
        CSVを読み込み、文字列を文字IDベクトルに変換してデータセットとして保持する
        CSV format:
            x : 入力文字列
            y : 教師文字列

        Parameters
        ----------
        source_csv : str or pathlib.Path
            読み込むCSVファイルのパス
        col_x : str, optional
            入力文字列のカラム名
        col_t : str, optional
            教師文字列のカラム名
        vocab_init : bool
            ボキャブラリー辞書を初期化する判定
        vocab_update : bool
            ボキャブラリー辞書を更新する判定

        Returns
        -------
        vec_x, vec_t : np.ndarray
            文字IDベクトル
            メンバとしても持つ
        '''
        # read CSV, split inputs and teachers
        df = pd.read_csv(source_csv)
        raw_x = [str(x) for x in df[col_x].values]
        raw_t = [str(t) for t in df[col_t].values]

        # create vocab dict
        # ついでに文字IDベクトルの次元数(=文字列の最大長さ)を取得する
        if vocab_init:
            self._init_vocab()
        dim_x = dim_t = 0
        for x, t in zip(raw_x, raw_t):
            if vocab_update:
                self._update_vocab(x)
                self._update_vocab(t)
            dim_x = len(x) if len(x) > dim_x else dim_x
            dim_t = len(t) if len(t) > dim_t else dim_t

        # convert from string to vector
        nb_data = len(df)
        vec_x = np.zeros((nb_data, dim_x), dtype=np.int)
        vec_t = np.zeros((nb_data, dim_t+2), dtype=np.int)  # +2 for SOS and EOS

        # 文字列長が最大長より短い場合、PADで埋める
        # ボキャブラリに存在しない文字はUNKで置き換える
        # 教師データはベクトル表現の前後にSOSとEOSを付ける
        sos, eos, pad = [self.__SOS], [self.__EOS], [self.__PAD]
        unk = self.char_to_id[self.__UNK]
        for i, (x, t) in enumerate(zip(raw_x, raw_t)):
            vec_x[i] = np.array([self.char_to_id.get(char, unk) for char in chain(x, pad * (dim_x - len(x)))])
            vec_t[i] = np.array([self.char_to_id.get(char, unk) for char in chain(sos, t, eos, pad * (dim_t - len(t)))])

        self.vec_x = vec_x
        self.vec_t = vec_t

        return self.vec_x, self.vec_t

    def _validate_args(self, x, t):
        '''
        引数のx, tが2階のndarrayであるかどうかを検査し、
        検査に通らなければself.vec_x, self.vec_tを代わりに返す

        Parameters
        ----------
        x : np.ndarray
            input data
        t : np.ndarray
            teacher data

        Returns
        -------
        np.ndarray, np.ndarray
            Arguments or member
        '''

        if not isinstance(x, np.ndarray) or x.ndim != 2:
            x = self.vec_x
        if not isinstance(t, np.ndarray) or t.ndim != 2:
            t = self.vec_t
        return x, t

    def to_csv(self, target_csv, x=None, t=None, col_x='x', col_t='y'):
        '''
        データセットをCSVに保存する
        CSV format:
            x__i : vec_x[i]
            y__i : vec_t[i]

        Parameters
        ----------
        target_csv : str or pathlib.Path
            保存するCSVファイルのパス
        x : np.ndarray, optional
            入力データ
            指定しない場合はself.vec_xを用いる
        t : np.ndarray, optional
            教師データ
            指定しない場合はself.vec_tを用いる
        col_x : str, optional
            入力データのカラム名prefix
        col_t : str, optional
            教師データのカラム名prefix

        Returns
        -------
        pd.DataFrame
            保存するCSVと同じフォーマットのDataFrame
        '''
        x, t = self._validate_args(x, t)

        columns_x = [col_x + '__' + str(i) for i in range(x.shape[1])]
        columns_t = [col_t + '__' + str(i) for i in range(t.shape[1])]

        df = pd.DataFrame(x, columns=columns_x).join(pd.DataFrame(t, columns=columns_t))
        df.to_csv(target_csv, index=False)

        return df

    def shuffle(self, x=None, t=None, seed=None):
        '''
        データをシャッフルする

        Parameters
        ----------
        x : np.ndarray, optional
            入力データ
            指定しない場合はself.vec_xを用いる
        t : np.ndarray, optional
            教師データ
            指定しない場合はself.vec_tを用いる
        seed : int, optional
            乱数シードの指定

        Returns
        -------
        np.ndarray, np.ndarray
            シャッフルされた入力データ、教師データ
        '''

        x, t = self._validate_args(x, t)
        nb_data = len(x)
        indices = np.arange(nb_data)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        x = x[indices]
        t = t[indices]

        return x, t

    def split_data(self, x=None, t=None, test_size=0.2, shuffle=True, seed=None):
        '''
        sklearn.model_selection.train_test_split()を呼び出して
        データセットを訓練データとテストデータに分割するヘルパー関数

        Returns
        ----------
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            x_train, x_test, t_train, t_test
        '''
        x, t = self._validate_args(x, t)

        return train_test_split(x, t, test_size=test_size, random_state=seed, shuffle=shuffle)

    def cv_dataset_gen(self, x=None, t=None, K=5):
        '''
        K-分割交差検証用のデータセットを返すジェネレータ
        データセットをK個に分割し、テストデータの位置を変えながら
        訓練データとテストデータを返す

        Parameters
        ----------
        x : np.ndarray, optional
            入力データ
        t : np.ndarray, optional
            教師データ
        K : int, optional
            データセットの分割数

        Yields
        ----------
        x_train, x_test, t_train, t_test : np.ndarray * 4
        '''

        x, t = self._validate_args(x, t)

        unit = x.shape[0] // K
        # kの値に応じてテストデータの位置を変える
        for i in range(K):
            x_train = np.r_[x[:unit*i], x[unit*(i+1):]]
            t_train = np.r_[t[:unit*i], t[unit*(i+1):]]
            x_test = x[unit*i:unit*(i+1)]
            t_test = t[unit*i:unit*(i+1)]
            yield x_train, x_test, t_train, t_test


    def result_to_csv(self, target_csv, vec_x, vec_t, vec_guess, cf=None, encoding='utf-8'):
        '''
        推論結果から文字列へ変換し、正解判定もつけてCSVへ保存する

        Parameters
        ----------
        target_csv : str or pathlib.Path
            結果を保存するCSVファイルのパス
        vec_x : np.ndarray
            入力データの文字IDベクトル表現
        vec_t : np.ndarray
            教師データの文字IDベクトル表現
        vec_guess : np.ndarray
            推論結果の文字IDベクトル表現
        cf : np.ndarray (N, )
            確信度
        encoding : str, optional
            文字コード
            データセットCSVと同じ文字コードにする

        Returns
        -------
        pd.DataFrame
            保存するCSVと同じフォーマットのDataFrame
        '''

        # 文字列へ変換
        xs = self.vec2seq(vec_x)
        ts = self.vec2seq(vec_t)
        gs = self.vec2seq(vec_guess)
        # レコードごとの正解判定
        correct = [1 if t == guess else 0 for t, guess in zip(ts, gs)]

        accuracy = sum(correct) / len(correct)
        print("Accuracy:", accuracy)
        if cf is None:
            df = pd.DataFrame({'x': xs, 'y': ts, 'guess': gs, 'correct': correct})
        else:
            df = pd.DataFrame({'x': xs, 'y': ts, 'guess': gs, 'correct': correct, 'cf': cf})
        df.to_csv(target_csv, index=False, encoding=encoding)

        return df

    def vec2seq(self, xs):
        '''
        Parameters
        ----------
        xs : np.ndarray  (ndim == 2)
            文字IDベクトルで表現された文字列データ
        Returns
        ----------
        list
            文字列のリスト
        '''
        if not isinstance(xs, np.ndarray) or xs.ndim != 2:
            raise ValueError('Argument "xs" is not word vector.')
        if GPU:
            xs = np.asnumpy(xs)
        # SOS, EOS, PADを除いてベクトル表現を文字に変換し、行ごとに連結して文字列のリストを返す
        return [''.join([self.id_to_char[x]\
                         for x in row if not self.id_to_char[x] in (self.__SOS, self.__EOS, self.__PAD)])\
                for row in xs]


    def seq2vec(self, x, size, vocab_update=False):
        '''
        テキストを文字IDベクトルへ変換する
        主に推論時に用いる

        Parameters
        ----------
        x : str
            翻訳元の入力テキスト
        size : int
            ベクトルの最大長
            不足分はPADでパディングされる
        vocab_update : bool, optional
            ボキャブラリーを更新するかどうか

        Returns
        -------
        np.ndarray (size, )
            文字IDベクトル
            ボキャブラリーに無い文字はUNKが充てられる
        '''

        if not isinstance(x, str):
            raise ValueError('Argument "x" is not string.')

        if vocab_update:
            self._update_vocab(x)

        pad = [self.__PAD]
        unk = self.char_to_id[self.__UNK]
        return np.array([self.char_to_id.get(char, unk) for char in chain(x[:size], pad * (size - len(x)))], dtype=np.int)


    def seqlist2vec(self, xs, size, vocab_update=False):
        '''
        テキストのリストを文字IDベクトルのミニバッチに変換するヘルパー関数

        Parameters
        ----------
        xs : list
            テキストのリスト
        size : int
            ベクトルの最大長
        vocab_update : bool, optional
            ボキャブラリ辞書を更新するかどうか (the default is False, which [default_description])

        Returns
        -------
        np.ndarray (len(xs), size)
            文字IDベクトルのミニバッチ
        '''

        if not isinstance(xs, list):
            raise ValueError('Argument "xs" is not list.')
        batch = []
        for x in xs:
            batch.append(self.seq2vec(x, size, vocab_update=vocab_update))

        return np.array(batch)
