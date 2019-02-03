# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset.sequence import TextSequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
from attention_seq2seq import AttentionSeq2seq
# from ch08.attention_seq2seq import AttentionSeq2seq

dataset_dir =Path('./dataset')
source_csv = dataset_dir / "addition.csv"
seq = TextSequence()
seq.read_csv(source_csv)
x_train, x_test, t_train, t_test = seq.split_data(seed=1, test_size=0.1)
char_to_id, id_to_char = seq.vocab

# Reverse input
is_reverse = True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# ハイパーパラメータ
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 15
max_grad = 5.0


# モデル選択
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
# model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)

image_path = 'result_' + model.__class__.__name__ + '.png'

optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(x_train, t_train, x_test, t_test,
            max_epoch=max_epoch,
            batch_size=batch_size,
            max_grad=max_grad)

trainer.plot(image_path)
# # Train
# acc_list = []
# loss_list = []
# for epoch in range(max_epoch):
#     trainer.fit(x_train, t_train,
#                 max_epoch=1,
#                 batch_size=batch_size, max_grad=max_grad)
#     loss_list.append(trainer.loss_list[-1])
#     correct_num = 0
#     for i in range(len(x_test)):
#         question, correct = x_test[[i]], t_test[[i]]
#         verbose = i < 10
#         correct_num += eval_seq2seq(model,
#                                     question,
#                                     correct,
#                                     id_to_char, verbose, is_reverse)
#     acc = float(correct_num) / len(x_test)
#     acc_list.append(acc)
#     print('val acc %.3f%%' % (acc * 100))

# グラフの描画
# x = np.arange(len(acc_list))
# plt.plot(x, acc_list, marker='o', label='Accuracy')
# plt.plot(x, loss_list, marker='x', label='Loss')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.ylim(0, 1.0)
# plt.legend()
# image_filename = 'result_seq2seq_attention.png'
# plt.savefig(image_filename)
# plt.show()
