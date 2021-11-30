#Usage: python classify.py <list of testing image> <list of training images>
import sys
import os
from utility_functions import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

testLst = sys.argv[1]
trainLst = sys.argv[2]

resultFile = open("test.clslbl", "w")

testFile = open(testLst,"r")
svmCat = ""
for query in testFile:
    query = query[:-1]
    print(query)
    KNN, queryDescriptor = getNeighbours(query , trainLst)
    sameClass, sameCat = checkSameClass(KNN)
    if(sameClass):
        resultFile.write(str(query + " " + sameCat + "\n"))
    else:
        trainData, trainLabels = getTrainingData(KNN)
        print(len(trainData), len(trainLabels))
        print("Training")
        mclass = OneVsRestClassifier(SVC()).fit(trainData, trainLabels)
        #testing of SVM
        testLabels = []
        lbl = query.split("/")[-2]
        for i in range(len(queryDescriptor)):
            testLabels += [lbl]
        predictedLabels = list(mclass.predict(queryDescriptor))
        svmCat = max(set(predictedLabels), key = predictedLabels.count)
        # print(predictedLabels)
        
        resultFile.write(str(query + " " + svmCat + "\n"))

testFile.close()       
resultFile.close()
