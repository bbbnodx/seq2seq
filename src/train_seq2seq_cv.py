# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data.sequence import TextSequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
from attention_seq2seq import AttentionSeq2seq

# ファイルパスの設定
dataset_dir =Path('../dataset')
source_csv = dataset_dir / "interpretation_train43_and_test40.csv"
result_dir = Path('../result')
dataset_name = source_csv.stem
model_dir = Path('../model')
encoding = 'utf-8'

# TextSequenceクラスからCSV読み込み
seq = TextSequence()
seq.read_csv(source_csv)
char_to_id, id_to_char = seq.vocab

# ハイパーパラメータ
vocab_size = len(char_to_id)
wordvec_size = 64
hidden_size = 128
batch_size = 32
max_epoch = 50
max_grad = 5.0

# shuffle
x, t = seq.shuffle(seed=1)

# K-分割交差法(test_size = 1/K)
K = 4

for i, (x_train, x_test, t_train, t_test) in enumerate(seq.cv_dataset_gen(x, t, test_size=1/K)):
    # モデル選択(パラメータの初期化)
    # model = Seq2seq(vocab_size, wordvec_size, hidden_size)
    model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
    # model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)
    print("Cross Validation: iter", str(i+1))
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

    # Save result as csv
    suffix = "_cv" + str(i + 1)
    result_train_csv = result_dir / ("result_" + dataset_name + "_" + model.__class__.__name__ + "_train"+ suffix +".csv")
    result_test_csv = result_dir / ("result_" + dataset_name + "_" + model.__class__.__name__ + "_test"+ suffix +".csv")

    seq.result_to_csv(result_train_csv, x_train, t_train, guess_train, encoding=encoding)
    seq.result_to_csv(result_test_csv, x_test, t_test, guess_test, encoding=encoding)

    # Plot learning curve and save it as png image
    image_path = result_dir / ('result_' + dataset_name + "_" + model.__class__.__name__ + suffix + '.png')
    trainer.plot(image_path)

    # release memory
    del model, optimizer, trainer
