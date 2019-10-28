from __future__ import print_function
from numpy import *
import numpy as np

def loadDataSet():
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
        return postingList, classVec

def creatVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #用于求两个集合的并
    return list(vocabSet) 

def setOfWords2Vec(vocabList,inputSet):#把句子变成m维向量（m为单词个数）
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 #
        else:
            print("the word:%s is not in my Vocabulary!"%word)
    print(returnVec)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)#侮辱文件出现的频率,计算先验
    p0Num = ones(numWords)#在好的邮件中每个词出现的次数
    p1Num = ones(numWords)#再坏的邮件中每个词出现的次数

    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    print("******")
    print(p1Denom)
    print(p0Denom)
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    list0Posts,listClasses = loadDataSet()#载入文本
    myVocabList = creatVocabList(list0Posts)#根据输入句子生成字典
    trainMat = []
    for postinDoc in list0Posts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))  
    p0v,p1v,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))

testingNB()