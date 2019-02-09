# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from data.sequence import TextSequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
from attention_seq2seq import AttentionSeq2seq

# ファイルパスの設定
dataset_dir =Path('../dataset')
# source_csv = dataset_dir / "addition.csv"
source_csv = dataset_dir / "interpretation_train43_and_test40.csv"
result_dir = Path('../result')
dataset_name = source_csv.stem
model_dir = Path('../model')
encoding = 'utf-8'

## 前処理
# TextSequenceクラスからCSV読み込み
seq = TextSequence()
seq.read_csv(source_csv)
char_to_id, id_to_char = seq.vocab

# ハイパーパラメータ
vocab_size = len(char_to_id)
wordvec_size = 128
hidden_size = 128
batch_size = 32
max_epoch = 100
max_grad = 5.0

# データセット分割
x_train, x_test, t_train, t_test = seq.split_data(seed=1, test_size=0.1)

# モデル選択
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

# Train
trainer.fit(x_train[:, ::-1], t_train, x_test[:, ::-1], t_test,
            max_epoch=max_epoch,
            batch_size=batch_size,
            max_grad=max_grad)

# Inference
start_id = seq.start_id
sample_size = seq.t_length
guess_train = model.generate(x_train[:, ::-1], start_id, sample_size)
guess_test = model.generate(x_test[:, ::-1], start_id, sample_size)



# 保存ファイルのファイル名生成
modelname = model.__class__.__name__
timestamp = datetime.now().strftime("_%y%m%d_%H%M")
save_dir = result_dir / (dataset_name + timestamp)
os.makedirs(save_dir, exist_ok=True)

# Save result as csv
result_train_csv = save_dir /  ("result_" + dataset_name + "_" + modelname + "_train.csv")
result_test_csv = save_dir / ("result_" + dataset_name + "_" + modelname + "_test.csv")
df_result_train = seq.result_to_csv(result_train_csv, x_train, t_train, guess_train, encoding=encoding)
df_result_test = seq.result_to_csv(result_test_csv, x_test, t_test, guess_test, encoding=encoding)

# Plot learning curve and save it as png image
image_path = save_dir / ('result_' + dataset_name + "_" + modelname + '.png')
trainer.plot(image_path=image_path)

# Save parameters
pickle_path = model_dir / (dataset_name + "_" + modelname + '_epoch' + max_epoch + "_" + timestamp + '.pkl')
model.save_params(file_name=pickle_path)
