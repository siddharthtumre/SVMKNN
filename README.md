# SVMKNN

## How to run

Extract the dataset Caltech 101

Create training and testing sets

`python create_train_test.py <path of caltech 101 folder> <no of training images per object category>`

For creating descriptors

`python create_descriptor.py <list of training images>`

For the final classification

`python classify.py <path of test.lst> <path of train.lst>`

The Output can be seen in the test.clslbl file