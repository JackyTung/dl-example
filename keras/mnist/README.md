# Sample - Run Mnist with Keras
The sample show how to run mnist with Keras on DeepQ platform

## Contents

1. [Basic usages](#basic-usages)
2. [How to prepare zip file of source code](#prepare-code)
3. [How to prepare zip file of source dataset](#prepare-dataset)

## Basic Usages

- train on the mnist dataset with number of epoch 10 and output model directory is output
```
python trainer.py --num-epoch=10 --outdir=output
```

## Prepare Code
- download code to local
```
git clone https://github.com/csigo/dl-example.git
```

- zip source code and the name is code.zip
```
cd dl-example/keras/mnist/
```
```
zip code.zip loader.py model.py trainer.py
```
- Choose the code (code.zip) when create training task in task management page

## Prepare Dataset
- download data to local from [download-link](http://yann.lecun.com/exdb/mnist/)
- gunzip all MNIST data files as the sample program does not deal with the uncompression
```
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
```
- zip source dataset and the name is data.zip
```
zip data.zip train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
```
- Upload the dataset (data.zip) in dataset management page

