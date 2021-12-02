# SVMKNN
This is an implementation of the paper [SVM-KNN: Discriminative Nearest Neighbor Classiﬁcation for Visual Category Recognition](https://ieeexplore.ieee.org/document/1641014)

We use the accurate distance function (discussed in the section Algorithm B.) on descriptors found using the **SIFT** descriptor for every image.
## How to run

Download and extract the dataset [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download)

Create training and testing sets

`python create_train_test.py <path of caltech 101 folder> <no of training images per object category>`

For creating descriptors

`python create_descriptor.py <list of training images>`

For the final classification

`python classify.py <path of test.lst> <path of train.lst>`

The Output can be seen in the test.clslbl file