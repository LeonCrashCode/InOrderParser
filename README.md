# InOrderParser

This implementation is based on the [cnn library](https://github.com/clab/cnn-v1) for this software to function. The reference paper is "In-Order Transition-based Constituent Parsing System"

#### Building

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make

#### Data
We borrow the code [get_oracle.py](https://github.com/clab/rnng/blob/master/get_oracle.py) to get top-down oracle
 
    ./get_oracle.py [training data in bracketed format] [training data in bracketed format] > [training top-down oracle]
    ./get_oracle.py [training data in bracketed format] [development data in bracketed format] > [development top-down oracle]   
    ./get_oracle.py [training data in bracketed format] [test data in bracketed format] > [test top-down oracle]

, and then compile pre2mid.cc to get pre2mid to convert them into in-order oracle

    g++ pre2mid.cc -o pre2mid
    ./pre2mid [training top-down oracle] > [training oracle]
    ./pre2mid [development top-down oracle] > [development oracle]
    ./pre2mid [test top-down oracle] > [test oracle]

If you want the related data, contact us.

#### Training

Ensure the related file are linked into the current directory.

    mkdir model/
    ./build/impl/Kparser --cnn-mem 1700 -x -T [training oracle] -d [development oracle] -C [development data in bracketed format] -P -t --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -D 0.2

#### Test
    
    ./build/impl/Kparser --cnn-mem 1700 -x -T [training oracle] -p [test oracle] -C [test data in bracketed format] -P --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -m [model file]

We provide the trained model file in [model](https://drive.google.com/file/d/0B1VhP65vISjoWmNjN0pfTmh5Vnc/view?usp=sharing)

#### Sampling

   ./build/impl/Kparser --cnn-mem 1700 -x -T [training oracle] -p [test oracle] -C [test data in bracketed format] -P --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -m [model file] --alpha 0.8 -s 100 > samples.act
   ./sample_mid2tree.py samples.act > samples.props

The samples.props could be feed into following reranking components. 

#### Contact

Jiangming Liu, jmliunlp@gmail.com
