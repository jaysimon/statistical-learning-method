#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 Guiyang Duocaibao S&T Co.,Ltd. All rights reserved.
# FileName : kNN.py_
# Author : Hou Wei
# Version : V1.0
# Date: 2019-10-18
# Description: kNN(nearest neighbour) practice with iris dataset
# dataset url: http://archive.ics.uci.edu/ml/datasets/Iris
# History:

import os
import numpy as np
import random


def prepare_iris_data():
    sDataPath = "../data/iris/iris.data"

    lList = open(sDataPath).read().splitlines()
    lList.pop()  # 删除空白行
    random.shuffle(lList)

    lTags = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    npX = np.zeros((len(lList), 4))
    npY = np.zeros((len(lList)))

    for i in range(len(lList)):
        sLine = lList[i]
        lWords = sLine.split(",")
        for j in range(4):
            npX[i, j] = float(lWords[j])
        npY[i] = lTags.index(lWords[4])
    np.save("../data/iris/npX.npy", npX)
    np.save("../data/iris/npY.npy", npY)
    return 0


class KNN():
    def __init__(self, npX, npTag):
        self.npX = npX
        self.npTag = npTag
        self.samplersCount = self.npX.shape[0]

    def compute_euclidean_distance(self, npA, npB):
        return np.sqrt(np.sum(np.square((npA - npB))))

    def infer(self, k, npInferX):
        lDist = []
        for i in range(self.samplersCount):
            fDist = self.compute_euclidean_distance(npInferX, self.npX[i])
            lDist.append(fDist)

        npRank = np.argsort(lDist)

        dcClass = {}
        for i in range(k):
            sKey = self.npTag[npRank[i]]
            if (sKey in dcClass):
                dcClass[sKey] += 1
            else:
                dcClass[sKey] = 1
        # print(max(dcClass))
        return max(dcClass)

    def test(self, k, npX, npY):
        iRightCount = 0
        for i in range(npX.shape[0]):
            npPred = self.infer(k, npX[i])
            if (npPred == npY[i]):
                iRightCount += 1
        print("[Log] K:%d Right:%d, Wrong:%d, Percentage:%.2f%%" %
              (k, iRightCount, npX.shape[0] - iRightCount,
               (iRightCount * 100.0) / npX.shape[0]))


def main():
    # 数据准备,执行一次即可
    # prepare_iris_data()
    npX = np.load("../data/iris/npX.npy")
    npY = np.load("../data/iris/npY.npy")

    fSplit = 0.9
    npTrainX = npX[:int(fSplit * npX.shape[0])]
    npTestX = npX[int(fSplit * npX.shape[0]):]
    npTrainY = npY[:int(fSplit * npX.shape[0])]
    npTestY = npY[int(fSplit * npX.shape[0]):]

    knn = KNN(npTrainX, npTrainY)
    knn.infer(4,npTestX[0])

    for i in range(1,10):
        knn.test(i, npTestX, npTestY)



if __name__ == "__main__":
    main()
