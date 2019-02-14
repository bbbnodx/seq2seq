# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class TextSequence:
    '''
    テキストシーケンスのクラス
    1行のテキストを文字単位に分割し、文字IDのベクトルに変換する
    また、CSVの入出力メソッドを持つ
    '''
    def __init__(self, EOS='_'):
        '''
        Parameters
        ----------
        EOS : str
            文章の終わりを示す特殊文字
            教師データの開始文字もこの文字にして、
            推論のときは、この文字から推論を始め、
            出力された文字を次の推論に用い……を繰り返す
        '''
        if not isinstance(EOS, str) or len(EOS) != 1:
            print("Argument 'EOS' is unexpected, so using '_' as 'EOS'.")
            EOS = '_'
        self.__EOS = EOS
        self._init_vocab()
        self.vec_x = None
        self.vec_t = None

    def _init_vocab(self):
        self.char_to_id = {self.__EOS: 0}
        self.id_to_char = {0: self.__EOS}

    @property
    def EOS(self):
        return self.__EOS

    @property
    def start_id(self):
        return self.char_to_id[self.__EOS]

    @property
    def x_length(self):
        return self.vec_x.shape[1]

    @property
    def t_length(self):
        return self.vec_t.shape[1] - 1

    @property
    def vocab(self):
        return self.char_to_id, self.id_to_char

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
        return self.vector_to_sequence(self.vec_x)

    @property
    def raw_t(self):
        '''
        文字列はデータとして持たず、ベクトル表現から変換して返す
        '''
        return self.vector_to_sequence(self.vec_t)

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


    def read_csv(self, source_csv, col_x='x', col_t='y', vocab_init=False):
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
            ボキャブラリーdictを初期化する判定

        Returns
        -------
        vec_x, vec_t : np.ndarray
            文字IDベクトル
            メンバとしても持つ
        '''
        # read CSV, split inputs and teachers
        # 教師データの文字列には先頭にEOSを付ける
        df = pd.read_csv(source_csv)
        raw_x = [str(x) for x in df[col_x].values]
        raw_t = [str(t) for t in df[col_t].values]

        # create vocab dict
        # ついでに文字IDベクトルの次元数(=文字列の最大長さ)を取得する
        if vocab_init:
            self._init_vocab()
        dim_x = dim_t = 0
        for x, t in zip(raw_x, raw_t):
            self._update_vocab(x)
            self._update_vocab(t)
            dim_x = len(x) if len(x) > dim_x else dim_x
            dim_t = len(t) if len(t) > dim_t else dim_t

        # convert from string to vector
        nb_data = len(df)
        vec_x = np.zeros((nb_data, dim_x), dtype=np.int)
        vec_t = np.zeros((nb_data, dim_t), dtype=np.int)
        for i, (x, t) in enumerate(zip(raw_x, raw_t)):
            vec_x[i] = [self.char_to_id[char] for char in x.ljust(dim_x, self.__EOS)]
            vec_t[i] = [self.char_to_id[char] for char in t.ljust(dim_t, self.__EOS)]

        # 教師データの先頭にEOSを付加する
        t_prefix = np.array([self.char_to_id[self.__EOS]], dtype=np.int).repeat(nb_data).reshape(nb_data, -1)
        self.vec_x = vec_x
        self.vec_t = np.c_[t_prefix, vec_t]

        return self.vec_x, self.vec_t

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
            指定しない場合はself.x_vecを用いる
        t : np.ndarray, optional
            教師データ
            指定しない場合はself.t_vecを用いる
        col_x : str, optional
            入力データのカラム名prefix
        col_t : str, optional
            教師データのカラム名prefix

        Returns
        -------
        pd.DataFrame
            保存するCSVと同じフォーマットのDataFrame
        '''
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            x = self.vec_x
        if not isinstance(t, np.ndarray) or t.ndim != 2:
            t = self.vec_t

        columns_x = [col_x + '__' + str(i) for i in range(x.shape[1])]
        columns_t = [col_t + '__' + str(i) for i in range(t.shape[1])]

        df = pd.DataFrame(x, columns=columns_x).join(pd.DataFrame(t, columns=columns_t))
        df.to_csv(target_csv, index=False)

        return df

    def shuffle(self, x=None, t=None, seed=None):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            x = self.vec_x
        if not isinstance(t, np.ndarray) or t.ndim != 2:
            t = self.vec_t
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
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            x = self.vec_x
        if not isinstance(t, np.ndarray) or t.ndim != 2:
            t = self.vec_t

        return train_test_split(x, t, test_size=test_size, random_state=seed, shuffle=shuffle)

    def cv_dataset_gen(self, x=None, t=None, test_size=0.2):
        '''
        交差検証用のデータセットを返すジェネレータ
        test_sizeに応じてデータセットを分割し、
        テストデータの位置を変えながら訓練データとテストデータを返す

        Parameters
        ----------
        x : np.ndarray, optional
            入力データ
        t : np.ndarray, optional
            教師データ
        test_size : float, optional
            テストデータのサイズ

        Yields
        ----------
        x_train, x_test, t_train, t_test : np.ndarray * 4
        '''

        if not isinstance(x, np.ndarray) or x.ndim != 2:
            x = self.vec_x
        if not isinstance(t, np.ndarray) or t.ndim != 2:
            t = self.vec_t

        unit = int(x.shape[0] * test_size)
        # kの値に応じてテストデータの位置を変える
        for k in range(int(1 / test_size)):
            x_train = np.r_[x[:unit*k], x[unit*(k+1):]]
            t_train = np.r_[t[:unit*k], t[unit*(k+1):]]
            x_test = x[unit*k:unit*(k+1)]
            t_test = t[unit*k:unit*(k+1)]
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
        xs = self.vector_to_sequence(vec_x)
        ts = self.vector_to_sequence(vec_t)
        gs = self.vector_to_sequence(vec_guess)
        # 正解判定
        correct = [1 if t == guess else 0 for t, guess in zip(ts, gs)]

        accuracy = sum(correct) / len(correct)
        print("Accuracy:", accuracy)
        if cf is None:
            df = pd.DataFrame({'x': xs, 'y': ts, 'guess': gs, 'correct': correct})
        else:
            df = pd.DataFrame({'x': xs, 'y': ts, 'guess': gs, 'correct': correct, 'cf': cf})
        df.to_csv(target_csv, index=False, encoding=encoding)

        return df

    def vector_to_sequence(self, xs):
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
        # EOS('_')を除いてベクトル表現を文字に変換し、行ごとに連結して文字列のリストを返す
        return [''.join([self.id_to_char[x]\
                         for x in row if not self.id_to_char[x] in (self.__EOS)])\
                for row in xs]
