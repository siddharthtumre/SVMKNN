import pickle
import cv2
import numpy as np

def saveDescriptorListFile(descriptor, descriptorFilename):
    file = open(descriptorFilename,"wb")
    pickle.dump(descriptor, file)
    file.close()    

def readDescriptorListFile(descriptorFilename):
    file = open(descriptorFilename,"rb")
    descriptorList = pickle.load(file)
    return descriptorList

def accurateDistance(aKeyPoints, A, bKeyPoints, B):  
    lamda = 0.25  
    r0 = 270 #avg image size of Caltech 101
    oneWayDistance = 0
    A = A.tolist()
    B = B.tolist()

    for i in range(len(A)):
        A_x = aKeyPoints[i].pt[0]
        A_y = aKeyPoints[i].pt[1]
        descriptorI = A[i]
        featureDistance = []

        for j in range (len(B)):
            B_x = bKeyPoints[j].pt[0]
            B_y = bKeyPoints[j].pt[1]
            descriptorJ = B [j]
            tmpDist = 0
            for k in range (len(descriptorJ)):
                tmpDist = tmpDist + (descriptorI[k] - descriptorJ[k])

            featureDistance +=[ (tmpDist)**2 + (lamda/r0)*(abs(A_x - B_x) + abs(A_y - B_y)) ]
        if(len(featureDistance) != 0):
            oneWayDistance = oneWayDistance + min(featureDistance)
    return oneWayDistance/len(A)

def getNeighbours(queryImage, trainLst):
    img = cv2.imread(queryImage)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    querykeypoints, queryDescriptor = sift.detectAndCompute(gray, None)
    distanceArray = []
    distanceImage = []

    trainFile = open(trainLst,"r")

    for image in trainFile:
        image = image[:-1]
        imageKeypoints = sift.detect(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY))
        imageDescriptorFile = image[:-4]  + '.decp'
        imageDescriptor = readDescriptorListFile (imageDescriptorFile)

        distance = accurateDistance(imageKeypoints, imageDescriptor,querykeypoints, queryDescriptor) + accurateDistance(querykeypoints, queryDescriptor, imageKeypoints, imageDescriptor)
        print(distance)
        # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # matches = bf.match(queryDescriptor,imageDescriptor)
        # print(len(queryDescriptor), len(imageDescriptor), len(matches))
        # matches = sorted(matches, key = lambda x:x.distance)

        distanceImage.append(image)
        distanceArray.append(distance)
        # print(len(matches))
        # print("queryid", "trainid", "imageid", "distance")
        # for i in range(len(matches)):
        #     print(matches[i].queryIdx, matches[i].trainIdx, matches[i].imgIdx, matches[i].distance)
        
    allNeighbours = list(zip(distanceImage, distanceArray))
    allNeighbours.sort(key = lambda x: x[1])

    trainFile.close()
    KNNs = []
    for i in allNeighbours[:30]:
        KNNs += [i[0]]
    return KNNs, queryDescriptor

def checkSameClass(KNN):
    cats = []
    cats += [KNN[0].split("/")[-2]]
    NN = KNN[1:]
    i = 0

    for image in NN:
        cats += [image.split("/")[-2]]
        i = i + 1
        if cats[0] != cats[i]:
            return False, cats[0]

    return True,cats[0]

def getTrainingData(KNN):
    data = []
    labels = []

    for neighbour in KNN:
        # print(neighbour)
        neighbour = neighbour[:-4]  + '.decp'
        temp = readDescriptorListFile(neighbour)
        temp = temp.tolist()
        data += temp
        cat = neighbour.split('/')[-2]
        for i in range(len(temp)):
            labels.append(cat)

    return data, labels