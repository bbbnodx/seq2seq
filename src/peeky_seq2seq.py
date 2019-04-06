# coding: utf-8
from common.time_layers import *
from common.base_model import BaseModel
from functools import partial
from seq2seq import Encoder, BiEncoder, Seq2seq, _init_parameter

class PeekyDecoder:
    '''
    PeekyDecoderレイヤ
    Encoderから引き継ぐ隠れ状態を全ての系列の入力、
    並びにLSTMの出力に結合する
        lstm_Wx.shape: (D+H, 4*H)
        affine_W.shape: (H+H, V)
    '''

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        # Decoder内部レイヤのパラメータを初期化する
        embed_W, _, lstm_Wh, lstm_b, _, affine_b = _init_parameter(V, D, H)
        rn = np.random.randn
        lstm_Wx = (rn(D+H, 4*H) / np.sqrt(D)).astype('f')
        affine_W = (rn(H+H, V) / np.sqrt(H)).astype('f')

        # レイヤの定義
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        # パラメータのセット
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs, h):
        '''
        Parameters
        ----------
        xs : np.ndarray(batch_size, seq_size)
            教師データts[:-1]
        h : np.ndarray(batch_size, hidden_size)
            Encoderの隠れ状態

        Returns
        -------
        np.ndarray
            推論のスコア
        '''
        N, T = xs.shape
        N, H = h.shape
        concat = partial(np.concatenate, axis=2)

        self.lstm.set_state(h)

        out = self.embed.forward(xs)  # (N, T, D)

        # Encoderから引き継いだ隠れ状態hを時系列分だけ拡張する
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        # LSTMへの入力を、hsとEmbedの出力を結合して用いる
        out = concat((hs, out))  # (N, T, H+D)
        out = self.lstm.forward(out)  # (N, T, H)
        # Affineへの入力も、hsとLSTMの出力を結合して用いる
        out = concat((hs, out))
        score = self.affine.forward(out)  # (N, T, V)

        # 隠れ層のサイズを保存する
        self.cache = H

        return score

    def backward(self, dscore):
        '''
        AffineレイヤとLSTMレイヤの逆伝播出力を
        xsとhsの分に分離する。
        それぞれのhsの和を取り、時系列方向に
        総和をとった配列をLSTMの隠れ状態hの勾配dhに
        加算してEncoderへ逆伝播する
        '''

        H = self.cache  # spliter

        dout = self.affine.backward(dscore)
        dhs0, dout = dout[:, :, :H], dout[:, :, H:]  # peeky
        dout = self.lstm.backward(dout)
        dhs1, dout = dout[:, :, :H], dout[:, :, H:]  # peeky
        self.embed.backward(dout)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + dhs.sum(axis=1)

        return dh

    def generate(self, h, start_id, sample_size):
        '''
        推論を実行し、文字IDベクトルを返す

        Parameters
        ----------
        h : ndarray
            Encoderから受け取る隠れ状態
        stard_id : int
            開始文字のID
        sample_size : int
            出力文字列の最大長さ

        Returns
        -------
        np.ndarray (N, sample_size)
            推論の結果
            長さsample_sizeの文字IDベクトル
        '''
        concat = partial(np.concatenate, axis=2)
        N = h.shape[0]
        sampled = []
        char_id = np.array(start_id).reshape(1, 1).repeat(N, axis=0)
        self.lstm.set_state(h)

        peeky_h = h.reshape(N, 1, -1)  # peeky
        for _ in range(sample_size):
            x = char_id
            out = self.embed.forward(x)
            out = concat((peeky_h, out))  # peeky
            out = self.lstm.forward(out)
            out = concat((peeky_h, out))  # peeky
            score = self.affine.forward(out)

            char_id = score.argmax(axis=2)
            sampled.append(char_id.flatten())

        return np.array(sampled, dtype=np.int).T

    def generate_with_cf(self, h, start_id, sample_size):
        '''
        推論を実行し、文字IDベクトルと確信度を返す

        Parameters
        ----------
        h : ndarray
            Encoderから受け取る隠れ状態
        stard_id : int
            開始文字のID
        sample_size : int
            出力文字列の最大長さ

        Returns
        -------
        np.ndarray (N, sample_size)
            推論の結果
            長さsample_sizeの文字IDベクトル
        np.ndarray (N, )
            系列データごとの確信度
        '''
        concat = partial(np.concatenate, axis=2)
        N = h.shape[0]
        sampled = []
        char_id = np.array(start_id).reshape(1, 1).repeat(N, axis=0)
        self.lstm.set_state(h)

        ### 確信度の取得用 ###
        sum_cf = np.zeros(N)
        counts = np.zeros(N, dtype=np.int)
        # cf_list = []
        # min_cf = np.ones(N)
        softmax = TimeSoftmax()
        ##########

        peeky_h = h.reshape(N, 1, -1)  # peeky
        for _ in range(sample_size):
            x = char_id
            out = self.embed.forward(x)
            out = concat((peeky_h, out))  # peeky
            out = self.lstm.forward(out)
            out = concat((peeky_h, out))  # peeky
            score = self.affine.forward(out)
            ### 確信度の取得 ###
            score = softmax.forward(score)
            ##########

            char_id = score.argmax(axis=2)
            sampled.append(char_id.flatten())

            ### 確信度の加算 ###
            mask = (char_id.flatten() != 0)
            sum_cf[mask] += score.max(axis=2).flatten()[mask]
            counts += mask

            # sum_cf += score.max(axis=2).flatten()
            # cf_list.append(score.reshape(N, -1))
            # if i < sample_size // 2:
            #     sum_cf += score.max(axis=2).flatten()
            # min_cf = np.minimum(min_cf, score.max(axis=2).flatten())
            ##########

        # cf = np.array(cf_list)  # (sample_size, N, V)
        cf = sum_cf / counts  # mean
        # cf = sum_cf / sample_size  # mean
        # cf = min_cf

        return np.array(sampled, dtype=np.int).T, cf


class PeekySeq2seq(Seq2seq):
    '''
    Peeky機構を用いたseq2seqモデル
    相違点はDecoderレイヤがPeekyDecoderレイヤに差し替わっただけである
    '''

    def __init__(self, vocab_size, wordvec_size, hidden_size, ignore_index=-1):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.loss = TimeSoftmaxWithLoss(ignore_index=ignore_index)

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

class PeekySeq2seqBiLSTM(Seq2seq):
    '''
    Encoderに双方向LSTMを採用したPeekySeq2seq
    '''

    def __init__(self, vocab_size, wordvec_size, hidden_size, ignore_index=-1):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = BiEncoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.loss = TimeSoftmaxWithLoss(ignore_index=ignore_index)

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
