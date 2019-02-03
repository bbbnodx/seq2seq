# coding: utf-8
from common.np import *  # import numpy as np
from common.layers import Softmax

class WeightSum:
    '''
    Encoderの全系列の隠れ状態hs(N, T, H)と
    系列ごとの重みを示すアライメントa(N, T)から積和を取り、
    現系列の変換に必要な情報を含むコンテキストベクトルc(N, H)を
    出力するレイヤ
    '''

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        '''
        重み係数aをnp.repeatで(N, T, H)に拡張し、
        hsとのアダマール積を取って時系列について総和を取ることで
        コンテキストベクトルc(N, H)を得る

        Parameters
        ----------
        hs : np.ndarray(N, T, H)
            Encoderの全系列の隠れ状態
        a : np.ndarray(N, T)
            系列ごとの重みを示すアライメント

        Returns
        -------
        np.ndarray(N, H)
            コンテキストベクトル
        '''
        N, T, H = hs.shape
        ar = a.reshape(N, T, 1)#.repeat(H, axis=2)
        t = hs * ar
        c = t.sum(axis=1)  # (N, H)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        '''
        sumの逆伝播はrepeat
        repeatの逆伝播はsum

        Parameters
        ----------
        dc : np.ndarray(N, H)
            コンテキストベクトルの勾配

        Returns
        -------
        dhs, da : np.ndarray(N, T, H), np.ndarray(N, T)
            全系列の隠れ状態hsの勾配と系列の重み係数aの勾配
        '''

        hs, ar = self.cache
        N, T, H = hs.shape

        # sumの逆伝播
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)  # (N, T, H)
        dhs = dt * ar
        dar = dt * hs

        # repeatの逆伝播
        da = dar.sum(axis=2)  # (N, T)

        return dhs, da

class AttentionWeight:
    '''
    Encoderの全系列の隠れ状態hs(N, T, H)と
    Decoderの現系列の隠れ状態h(N, H)とのドット積をとり、
    softmax関数にかけることで系列ごとのアライメントa(N, T)を
    出力するレイヤ
    '''

    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        '''
        Decoderの隠れ状態h(N, H)をnp.repeatで(N, T, H)に拡張し、
        hsとのアダマール積を取ってHについて総和を取り、
        Softmax関数で正規化してアライメントa(N, T)を得る

        Parameters
        ----------
        hs : np.ndarray(N, T, H)
            Encoderの全系列の隠れ状態
        h : np.ndarray(N, H)
            Decoderの現系列の隠れ状態

        Returns
        -------
        np.ndarray(N, T)
            hsに対し、系列ごとの重みを示すアライメント
        '''

        N, T, H = hs.shape

        hr = h.reshape(N, 1, H)#.repeat(T, axis=1)
        t = hs * hr  # (N, T, H)
        s = t.sum(axis=2)
        a = self.softmax.forward(s)  # (N, T)

        self.cache = (hs, hr)
        return a

    def backward(self, da):
        '''
        sumの逆伝播はrepeat
        repeatの逆伝播はsum

        Parameters
        ----------
        da : np.ndarray(N, T)
            アライメントの勾配

        Returns
        -------
        dhs, dh : np.ndarray(N, T, H), np.ndarray(N, H)
            全系列の隠れ状態hsの勾配と系列の隠れ状態hの勾配
        '''
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr  # (N, T, H)
        dhr = dt * hs  # (N, T, H)
        dh = dhr.sum(axis=1)  # (N, H)

        return dhs, dh

class Attention:
    '''
    Attentionレイヤ
        in - AttentionWeight - WeightSum - out

    Input
    -------
    hs, h : np.ndarray(N, T, H), np.ndarray(N, H)
        Encoderの全系列の隠れ状態, Decoderの現系列隠れ状態

    Output
    -------
    c : np.ndarray(N, H)
        現系列の変換に必要な情報を含むコンテキストベクトル
    '''

    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        # 外から各系列の重みを参照できるようにする
        self.__attention_weight = None

    @property
    def attention_weight(self):
        return self.__attention_weight

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        c = self.weight_sum_layer.forward(hs, a)

        self.__attention_weight = a
        return c

    def backward(self, dc):
        dhs0, da = self.weight_sum_layer.backward(dc)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1

        return dhs, dh

class TimeAttention:
    '''
    Attentionレイヤの全系列バージョン

    Input
    -------
    hs_enc, hs_dec : np.ndarray(N, T, H), np.ndarray(N, T, H)
        Encoderの全系列の隠れ状態, Decoderの全系列の隠れ状態

    Output
    -------
    cs : np.ndarray(N, T, H)
        変換に必要な情報を含むコンテキストベクトルの全系列分
    '''

    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t, layer in enumerate(self.layers):
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec
