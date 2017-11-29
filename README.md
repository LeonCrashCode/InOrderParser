# InOrderParser

This implementation is based on the [cnn library](https://github.com/clab/cnn-v1) for this software to function. The reference paper is "In-Order Transition-based Constituent Parsing System", which is accepted by Tansactions of ACL.  The system achieves the state-of-the-art results on the standard benchmark PTB and CTB 5.1 by obtaining 91.8 F1 and 86.1 F1, respectively. With sampling and reranking strategy, it achieves 94.2 F1 and 88.0 F1, respectively. By converting to dependencies, it achieves 96.2 UAS (95.2 LAS) and 89.4 UAS (88.4 LAS), respectively. On single i7 CPU, the speed is 60 sentence per second. 

## Building
The boost version is 1.5.4.

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make

There are two implementations, Kparser and Kparser-standard. Kparser is used for standard experiments, while Kparser-standard is easy-use.

## Experiments

#### Data

You could use the scripts to convert the format of training, development and test data, respectively.

    python ./scripts/get_oracle.py [en|ch] [training data in bracketed format] [training data in bracketed format] > [training oracle]
    python ./scripts/get_oracle.py [en|ch] [training data in bracketed format] [development data in bracketed format] > [development oracle]   
    python ./scripts/get_oracle.py [en|ch] [training data in bracketed format] [test data in bracketed format] > [test oracle]

If you require the related data, contact us.

#### Training

Ensure the related file are linked into the current directory.

    mkdir model/
    ./build/impl/Kparser --cnn-mem 1700 --training_data [training oracle] --dev_data [development oracle] --bracketing_dev_data [development data in bracketed format] -P -t --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -D 0.2

#### Test
    
    ./build/impl/Kparser --cnn-mem 1700 --training_data [training oracle] --test_data [test oracle] --bracketing_dev_data [test data in bracketed format] -P --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -m [model file]

The automatically generated file test.eval is the result file.

We provide the trained models: [English model](https://drive.google.com/file/d/0B1VhP65vISjoWmNjN0pfTmh5Vnc/view?usp=sharing) and pretrained word embeddings [sskip.100.vectors](https://drive.google.com/open?id=0B1VhP65vISjoZ3ppTnR3YXRMd1E) for English; [Chinese model](https://drive.google.com/open?id=0B1VhP65vISjoVjZKT2U1amFXVGc) and pretrained word embeddings [zzgiga.sskip.80.vectors](https://drive.google.com/open?id=0B1VhP65vISjoeGJsX2syOGhLWnc) for Chinese

#### Sampling

    ./build/impl/Kparser --cnn-mem 1700 --training_data [training oracle] --test_data [test oracle] --bracketing_dev_data [test data in bracketed format] -P --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -m [model file] --alpha 0.8 -s 100 > samples.act
    ./mid2tree.py samples.act > samples.trees

The samples.props could be fed into following reranking components. 

## Easy Usage

Download the [English model](https://drive.google.com/open?id=0B1VhP65vISjoSXRHelVnSVNYSjA) and the [Chinese model](https://drive.google.com/open?id=0B1VhP65vISjodDM2NW9vRFdOQmM).

    ./build/impl/Kparser-standard --cnn-mem 1700 --model_dir [model directory] -w [pretrained word embeddings] --train_dict [model directory]/train_dict --lang [en/ch] < [stdin] > [stdout]

The standard input should follow the fomart, Word1 POS1 Word2 POS2 ... Wordn POSn. The example is

    No RB , , it PRP was VBD n't RB Black NNP Monday NNP . .

The standard output is tree in bracketed format.

    (S (INTJ (RB No)) (, ,) (NP (PRP it)) (VP (VBD was) (RB n't) (NP (NNP Black) (NNP Monday))) (. .)) 

If you want to sample trees, you should added --samples [number of samples] --a [alpha], for example, --samples 100 --a 0.8

## Citation

    @article{TACL1199,   
        author = {Liu, Jiangming  and Zhang, Yue },   
        title = {In-Order Transition-based Constituent Parsing},   
        journal = {Transactions of the Association for Computational Linguistics},   
        volume = {5},   
        year = {2017},   
        issn = {2307-387X},   
        pages = {413--424}   
        }

## Contact

Jiangming Liu, jmliunlp@gmail.com

Yue Zhang, yue_zhang@sutd.edu.sg
