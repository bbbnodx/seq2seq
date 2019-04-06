<!--
 Copyright (c) 2019 flexfirm
-->

# 品種翻訳
LSTMを用いたseq2seqモデルで品種翻訳を学習する深層学習モデルです。

## 目次
- [ディレクトリ構成](#ディレクトリ構成)
- [開発環境](#開発環境)
- [プログラムの実行](#プログラムの実行)
  - [学習の実行](#学習の実行)
  - [推論の実行](#推論の実行)
- [入出力ファイルの形式](#入出力ファイルの形式)
- [主要クラス](#主要クラス)
  - [Seq2seq(src/seq2seq.py)](#seq2seqsrcseq2seqpy)
  - [TextSequence(src/data/sequence.py)](#textsequencesrcdatasequencepy)
  - [Trainer(src/common/trainer.py)](#trainersrccommontrainerpy)

## ディレクトリ構成
```
/
├── README.md
├── dataset/              <- 学習に用いるデータセット
│   └── README.md         <- README for dataset.
├── models/               <- 訓練済みモデルファイル
├── notebooks/            <- Jupyter notebooks.
├── results/              <- 品種翻訳結果ファイルの出力先
├── requirements.txt      <- 必要パッケージ
└── src
    ├── common/           <- 基本レイヤや関数など
    ├── data/             <- データ操作モジュール
    ├── seq2seq.py        <- Seq2seqモデルクラス
    └── train_seq2seq.py  <- Seq2seqの学習を実行するスクリプト
```

## 開発環境
Python 3.6.8  
パッケージは以下の通りです。  
```
numpy==1.14.3
pandas==0.23.4
matplotlib==3.0.2
scikit-learn==0.20.1
jupyter==1.0.0
```
必要パッケージは以下のコマンドで一括インストールできます。
```bash
$ pip install -r requirements.txt
```

## プログラムの実行
### 学習の実行
基本的にはJupyter Notebookで実行します。
```bash
$ jupyter notebook
```

[notebooks/train_seq2seq.ipynb](./notebooks/train_seq2seq.ipynb)を開いて、上から順にコードを実行すれば学習できます。  
ハイパーパラメータや入出力ファイル名などは適宜変更してください。  
実行結果として、CSVと学習曲線のPNG画像を`results/`以下に保存し、  
学習したパラメータを`models/`以下に保存します。

CLIから実行する場合は、srcディレクトリに移動してから`train_seq2seq.py`を実行してください。
```bash
$ cd ./src
$ python train_seq2seq.py
```
モデル選択、エポック数、ハイパーパラメータなどはハードコーディングされています。

### 推論の実行
学習と同様に、Jupyter Notebookから推論を実行できます。  
[notebooks/inference_seq2seq.ipynb](./notebooks/inference_seq2seq.ipynb)を開いて、実行してください。  

このとき、学習済みのパラメータを保存したpickleファイルが必要です。  
pickleファイルが無いときは、先に学習を実行してください。  
また、読み込むpickleファイルはハイパーパラメータが使用するモデルと一致している必要があります。  
pickleファイル名を参照し、適切なpickleファイルを読み込んでください。


## 入出力ファイルの形式
データセットは以下のCSV形式で準備して、datasetディレクトリに配置してください。  
1行目はカラム名を表し、xが入力データ、yが教師データです：
```csv
x,y
foo,bar
baz,qux
...
```

推論後の出力ファイルは`results/`以下に"データセット名+タイムスタンプ"のディレクトリを作成し、  
その中にCSVと画像ファイルを保存します。  
追加されたカラムのguessが推論結果、correctが正解判定(正解が1)です：
```csv
x,y,guess,correct
foo,bar,bar,1
baz,qux,foobar,0
...
```

## 主要クラス
### Seq2seq([src/seq2seq.py](./src/seq2seq.py))
Seq2seqのモデルクラスです。  
`PeekySeq2seq`などの派生モデルの基底クラスでもあります。
- `__init__(vocab_size, wordvec_size, hidden_size)` : ハイパーパラメータを指定します
- `forward(xs, ts)` : 順伝搬を実行し、損失関数の値を返します
- `backward()` : 逆伝搬を実行し、勾配を更新します
- `generate(xs, start_id, sample_size)` : 推論のため、教師データを用いずに出力を生成して返します
- `generate_with_cf(xs, start_id, sample_size)` : 推論結果に加え、確信度を取得して返します
- `validate(xs, ts)` : generate()を実行し、教師データと比較して正解数を返します


### TextSequence([src/data/sequence.py](./src/data/sequence.py))
データセットの入出力を扱うクラスです。  
CSVの入出力、ボキャブラリの構築などを行います。
- `read_csv(source_csv)` : CSVファイルを読み込み、文字IDベクトル表現に変換して記憶します
- `split_data(x=None, t=None, test_size=0.2)` : データセットを分割して返します。
- `result_to_csv(target_csv, vec_x, vec_t, vec_guess)` : 推論結果を文字列へ変換し、CSVに保存します


### Trainer([src/common/trainer.py](./src/common/trainer.py))
学習を実行するためのクラスです。  
ミニバッチ学習、学習過程の表示、学習曲線の可視化などを行います。  
- `__init__(model, optimizer)` : モデルと最適化関数を設定します
- `fit(x_train, t_train, x_test, t_test, max_epoch)` : 学習を実行します
- `plot(image_path=None)` : fit()で記録した学習曲線をプロットしてグラフを表示します
