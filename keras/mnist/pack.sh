#!/usr/bin/env bash

prepare_code () {
 zip code.zip loader.py model.py trainer.py
}

prepare_dataset () {
 echo "Download the mnist dataset"
 wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
 wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
 wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
 wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

 echo "Zip source dataset and the name is data.zip"
 zip data.zip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
}

prepare_code
prepare_dataset

echo "Both code.zip and data.zip are already prepared !"
