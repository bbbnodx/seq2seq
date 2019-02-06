# coding: utf-8
from common.time_layers import *
from common.base_model import BaseModel
from functools import reduce

def _init_parameter(V, D, H):
    '''
    パラメータ初期値を生成して返す

    Parameters
    ----------
    V : int
        size of vocabulary
    D : int
        size of word vector
    H : int
        size of hidden layer

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        embed_W: Embedレイヤ
        lstm_Wx, lstm_Wh, lstm_b: LSTMレイヤ
        affine_W, affine_b: Affineレイヤ
    '''

    rn = np.random.randn
    embed_W = (rn(V, D) / 100).astype('f')
    lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
    lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    lstm_b = np.zeros(4*H).astype('f')
    affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
    affine_b = np.zeros(V).astype('f')

    return embed_W,\
           lstm_Wx, lstm_Wh, lstm_b,\
           affine_W, affine_b

class Encoder:
    '''
    Encoderレイヤ
    通常のLSTMとほぼ同じだが、順伝播の出力として
    最後の隠れ状態を返すところが異なる
    '''

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        # Encoder内部レイヤのパラメータを初期化する
        embed_W, lstm_Wx, lstm_Wh, lstm_b, _, _ = _init_parameter(V, D, H)

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        # 最後の隠れ状態を返す
        return hs[:, -1, :]

    def backward(self, dh):
        # 勾配の初期化
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)

        return dout

class Decoder:
    '''
    Decoderレイヤ
    Encoderの隠れ状態を引き継いで、
    現在の文字から次の文字を出力する。
    学習時は教師データを用いて並列処理できるが、
    推論時は開始文字のみを与えて順に伝播する必要がある
    学習時と推論時で異なる処理が必要なため、
    順伝播の処理をforward()とgenerate()に分けている
    '''

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        # Decoder内部レイヤのパラメータを初期化する
        embed_W, lstm_Wx, lstm_Wh, lstm_b, affine_W, affine_b = _init_parameter(V, D, H)

        # レイヤの定義
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        # パラメータのセット
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        '''
        Parameters
        ----------
        xs : np.ndarray
            教師データts[:-1]
        h : np.ndarray
            Encoderの隠れ状態

        Returns
        -------
        np.ndarray
            推論のスコア
        '''

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh

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

        sampled = []
        N = h.shape[0]
        char_id = np.array(start_id).reshape(1, 1).repeat(N, axis=0)
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = char_id
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            char_id = score.argmax(axis=2)
            sampled.append(char_id.flatten())

        return np.array(sampled).T

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

        sampled = []
        N = h.shape[0]
        char_id = np.array(start_id).reshape(1, 1).repeat(N, axis=0)
        self.lstm.set_state(h)

        ### 確信度の取得用 ###
        sum_cf = 0
        softmax = TimeSoftmax()
        ##########

        for _ in range(sample_size):
            x = char_id
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)
            ### 確信度の取得 ###
            score = softmax.forward(score)
            ##########

            char_id = score.argmax(axis=2)
            sampled.append(char_id.flatten())

            ### 確信度の加算 ###
            sum_cf += score.max(axis=2).flatten()
            ##########

        cf = sum_cf / sample_size  # mean

        return np.array(sampled).T, cf


class Seq2seq(BaseModel):
    '''
    Seq2seqモデルのクラス
    改良モデルの基底クラスでもあり、__init__以外のメソッドは共用する
    '''

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.loss = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        # Decoderの入力データ:教師データの最終文字を除いたもの
        # 損失関数の教師データ:教師データの2文字目以降
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.loss.forward(score, decoder_ts)

        return loss

    def backward(self, dout=1):
        dout = self.loss.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)

        return dout

    def generate(self, xs, start_id, sample_size):
        '''
        入力データから推論を行い、推論結果の文字IDベクトルを得る

        Parameters
        ----------
        xs : np.ndarray (N, input_size)
            入力データのミニバッチ
        start_id : int
            開始文字ID
        sample_size : int
            出力文字列の長さ

        Returns
        -------
        np.ndarray (N, sample_size)
            推論結果の文字IDベクトルのミニバッチ
        '''

        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)

        return sampled

    def generate_with_cf(self, xs, start_id, sample_size):
        '''
        入力データから推論を行い、推論結果の文字IDベクトルと確信度を得る

        Parameters
        ----------
        xs : np.ndarray (N, input_size)
            入力データのミニバッチ
        start_id : int
            開始文字ID
        sample_size : int
            出力文字列の長さ

        Returns
        -------
        np.ndarray (N, sample_size)
            推論結果の文字IDベクトルのミニバッチ
        np.ndarray (N, )
            シーケンスごとの確信度
        '''

        h = self.encoder.forward(xs)
        sampled, cf = self.decoder.generate_with_cf(h, start_id, sample_size)

        return sampled, cf

    def validation(self, xs, ts):
        '''
        xsから推論を行い、tsと比較して正解数を返す

        Parameters
        ----------
        xs : np.ndarray
            入力のミニバッチ
        ts : np.ndarray
            教師データのミニバッチ

        Returns
        -------
        int
            バッチデータにおける正解数
        '''

        start_id = ts[0, 0]
        correct = ts[:, 1:]
        sample_size = correct.shape[1]

        guess = self.generate(xs, start_id, sample_size)
        # データごとの正解判定を得る
        # 文字単位で正解判定を行い、行について論理積で畳み込んで系列単位の正解判定を得る
        valid_sequence = reduce(np.logical_and, (correct == guess).T)
        # 総和をとってバッチデータの正解数を返す
        return valid_sequence.sum()
