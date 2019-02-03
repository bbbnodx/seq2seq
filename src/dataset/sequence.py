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
    def __init__(self, t_prefix='_'):
        '''
        Parameters
        ----------
        t_prefix : str
            教師データの開始文字
            推論のときは、この文字から推論を始め、
            出力された文字を次の推論に用い……を繰り返す
        '''
        self._init_vocab()
        self.vec_x = None
        self.vec_t = None
        if not isinstance(t_prefix, str) or len(t_prefix) != 1:
            print("Argument 't_prefix' is unexpected, so using '_' as 't_prefix'.")
            t_prefix = '_'
        self.__t_prefix = t_prefix

    @property
    def t_prefix(self):
        return self.__t_prefix

    @property
    def vocab(self):
        return self.char_to_id, self.id_to_char

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

    def _init_vocab(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self._update_vocab(' ')  # 空白をid=0で登録

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


    def read_csv(self, source_csv, col_x='x', col_t='y'):
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

        Returns
        -------
        vec_x, vec_t : np.ndarray
            文字IDベクトル
            メンバとしても持つ
        '''
        # read CSV, split inputs and teachers
        # 教師データの文字列には先頭に開始文字t_prefixを付ける
        df = pd.read_csv(source_csv)
        raw_x = [str(x) for x in df[col_x].values]
        raw_t = [self.t_prefix + str(t) for t in df[col_t].values]

        # create vocab dict
        # ついでに文字IDベクトルの次元数(=文字列の最大長さ)を取得する
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
            vec_x[i] = [self.char_to_id[char] for char in x.ljust(dim_x, ' ')]
            vec_t[i] = [self.char_to_id[char] for char in t.ljust(dim_t, ' ')]

        self.vec_x = vec_x
        self.vec_t = vec_t

        return vec_x, vec_t

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

    def result_to_csv(self, target_csv, vec_x, vec_t, vec_guess):
        # 文字列へ変換
        xs = self.vector_to_sequence(vec_x)
        ts = self.vector_to_sequence(vec_t)
        gs = self.vector_to_sequence(vec_guess)
        # 正解判定
        correct = [1 if t == guess else 0 for t, guess in zip(ts, gs)]

        accuracy = sum(correct) / len(correct)
        print("Accuracy:", accuracy)

        df = pd.DataFrame({'x': xs, 'y': ts, 'guess': gs, 'correct': correct})
        df.to_csv(target_csv, index=False)

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
        # 空白と開始文字('_')を除いてベクトル表現を文字に変換し、行ごとに連結して文字列のリストを返す
        return [''.join([id_to_char[x]\
                         for x in row if not id_to_char[x] in (' '+self.__t_prefix)])\
                for row in xs]
