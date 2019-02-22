<!--
 Copyright (c) 2019 bbbnodx

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Seq2seq
Sequence to Sequence learning model with LSTM.


### The resulting directory structure
---
```
/
├── LICENSE
├── README.md             <- The top-level README for developers using this project.
├── dataset/              <- Dataset for seq2seq learning.
├── models/               <- Trained and serialized models.
├── notebooks/            <- Jupyter notebooks.
├── results/              <- Result CSV and plotting learning curve figure.
├── requirements.txt      <- The requirements file.
└── src
    ├── common/           <- Scripts for common functions, layers, and others.
    ├── data/             <- Scripts to data wrangling.
    ├── seq2seq.py        <- Scripts to seq2seq model.
    └── train_seq2seq.py  <- Scripts to training seq2seq.
```


### Installing development requirements
---
```bash
pip install -r requirements.txt
```

### Training
---
```bash
cd ./src
python train_seq2seq.py
```

or, open `notebooks/train_seq2seq.ipynb` on Jupyter Notebook.