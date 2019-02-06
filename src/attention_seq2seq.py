from common.time_layers import *
from seq2seq import Encoder, Seq2seq
from attention_layer import TimeAttention

class AttentionEncoder(Encoder):
    '''
    Encoderからの変更点は出力が最後の隠れ状態hから
    全系列の隠れ状態hsになったことのみ
    '''

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)

        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)

        return dout

class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        # Decoder内部レイヤのパラメータを初期化する
        embed_W, lstm_Wx, lstm_Wh, lstm_b, affine_W, affine_b = _init_parameter_attention(V, D, H)

        # レイヤの定義
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        # パラメータのセット
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, hs_enc):
        h = hs_enc[:, -1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs_dec = self.lstm.forward(out)
        cs = self.attention.forward(hs_enc, hs_dec)
        out = np.concatenate((cs, hs_dec), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dcs, dhs_dec0 = dout[:, :, :H], dout[:, :, H:]
        dhs_enc, dhs_dec1 = self.attention.backward(dcs)
        dhs_dec = dhs_dec0 + dhs_dec1
        dout = self.lstm.backward(dhs_dec)
        dh = self.lstm.dh
        dhs_enc[:, -1] += dh
        self.embed.backward(dout)

        return dhs_enc

    def generate(self, hs_enc, start_id, sample_size):
        N= hs_enc.shape[0]
        h = hs_enc[:, -1]
        self.lstm.set_state(h)
        sampled = []
        char_id = np.array(start_id).reshape(1, 1).repeat(N, axis=0)

        for _ in range(sample_size):
            x = char_id
            out = self.embed.forward(x)
            hs_dec = self.lstm.forward(out)
            cs = self.attention.forward(hs_enc, hs_dec)
            out = np.concatenate((cs, hs_dec), axis=2)
            score = self.affine.forward(out)

            char_id = score.argmax(axis=2)
            sampled.append(char_id.flatten())

        return np.array(sampled).T

    def generate_with_cf(self, hs_enc, start_id, sample_size):
        N= hs_enc.shape[0]
        h = hs_enc[:, -1]
        self.lstm.set_state(h)
        sampled = []
        char_id = np.array(start_id).reshape(1, 1).repeat(N, axis=0)

        ### 確信度の取得用 ###
        sum_cf = 0
        softmax = TimeSoftmax()
        ##########

        for _ in range(sample_size):
            x = char_id
            out = self.embed.forward(x)
            hs_dec = self.lstm.forward(out)
            cs = self.attention.forward(hs_enc, hs_dec)
            out = np.concatenate((cs, hs_dec), axis=2)
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

class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.loss = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads


def _init_parameter_attention(V, D, H):
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
    affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
    affine_b = np.zeros(V).astype('f')

    return embed_W,\
        lstm_Wx, lstm_Wh, lstm_b,\
        affine_W, affine_b
