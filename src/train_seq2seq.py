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
# source_csv = dataset_dir / "addition.csv"
source_csv = dataset_dir / "interpretation_train43_and_test40.csv"
result_dir = Path('../result')
dataset_name = source_csv.stem
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
wordvec_size = 64
hidden_size = 128
batch_size = 32
max_epoch = 100
max_grad = 5.0


# モデル選択
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
# model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)


optimizer = Adam()
trainer = Trainer(model, optimizer)

# Train
trainer.fit(x_train, t_train, x_test, t_test,
            max_epoch=max_epoch,
            batch_size=batch_size,
            max_grad=max_grad)

# Inference
start_id = seq.start_id
sample_size = seq.t_length
guess_train = model.generate(x_train, start_id, sample_size)
guess_test = model.generate(x_test, start_id, sample_size)
# guess_train, sum_cf_train = model.generate(x_train, start_id, sample_size)
# guess_test, sum_cf_test = model.generate(x_test, start_id, sample_size)


# Save result as csv
encoding = 'utf-8'
result_train_csv = result_dir / ("result_" + dataset_name + "_" + model.__class__.__name__ + "_train.csv")
result_test_csv = result_dir / ("result_" + dataset_name + "_" + model.__class__.__name__ + "_test.csv")
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]  # 逆順を元の順に戻す
seq.result_to_csv(result_train_csv, x_train, t_train, guess_train, encoding=encoding)
seq.result_to_csv(result_test_csv, x_test, t_test, guess_test, encoding=encoding)

# Plot learning curve and save it as png image
image_path = result_dir / ('result_' + dataset_name + "_" + model.__class__.__name__ + '.png')
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
