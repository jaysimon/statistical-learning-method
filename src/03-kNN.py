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
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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


class Drawer():
    def maxDepth(self, root):
        if root == None:
            return 0
        # print(root.val)
        iLeft = self.maxDepth(root.left)
        iRight = self.maxDepth(root.right)
        iMax = iLeft if (iLeft > iRight) else iRight
        return iMax + 1

    def draw_node(self, ax, tRoot, iX, iY):
        if tRoot is None:
            return None
        iDepth = self.maxDepth(tRoot)
        cir = Circle(xy=(iX, iY), radius=1.5, alpha=0.5)
        ax.add_patch(cir)
        plt.text(iX - 0.9, iY - 0.3, str(tRoot.val), fontsize=10)

        if (tRoot.left):
            plt.plot([iX - 1, iX - 2 * iDepth + 1], [iY - 1, iY - 4 + 1], "b")
        if (tRoot.right):
            plt.plot([iX + 1, iX + 2 * iDepth - 1], [iY - 1, iY - 4 + 1], "b")
        self.draw_node(ax, tRoot.left, iX - 2 * iDepth, iY - 4)
        self.draw_node(ax, tRoot.right, iX + 2 * iDepth, iY - 4)

    def draw_tree(self, tRoot):
        fig = plt.figure(1, facecolor="white")
        fig.clf()
        ax = fig.add_subplot(111)
        self.draw_node(ax, tRoot, 0, 8)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.show()


class TreeNode(object):
    def __init__(self, npX, iSplitIndex):
        self.val = npX
        self.left = None
        self.right = None
        self.iSplit = iSplitIndex


class KNN():
    def __init__(self, npX, npTag):
        self.npX = npX
        self.npTag = npTag
        self.samplersCount = self.npX.shape[0]
        iKFeature = 0
        self.tRoot = self.create_kdtree(iKFeature, npX)
        drawer = Drawer()
        drawer.draw_tree(self.tRoot)

    def create_kdtree(self, iKFeature, npX):
        if npX.shape[0] == 0:
            return None
        npX = npX[npX[:, iKFeature].argsort()]
        npXLeft = npX[:int(npX.shape[0] / 2)]
        npXRight = npX[int(npX.shape[0] / 2) + 1:]
        tRoot = TreeNode(npX[int(npX.shape[0] / 2)], iKFeature)
        # print(npX[int(npX.shape[0] / 2)])
        tRoot.left = self.create_kdtree((iKFeature + 1) % npX.shape[1], npXLeft)
        tRoot.right = self.create_kdtree((iKFeature + 1) % npX.shape[1],
                                         npXRight)
        return tRoot

    def find_nearest(self, npX, tRoot, tPoint, fDist):
        if (tRoot.left == tRoot.right == None):
            return None

    def compute_distance(self, npA, npB):
        return np.sqrt(np.sum(np.square((npA - npB))))

    def infer(self, k, npInferX):
        lDist = []
        for i in range(self.samplersCount):
            fDist = self.compute_distance(npInferX, self.npX[i])
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

    npTrainX = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    npTrainY = np.array([1, 1, 2, 2, 3, 3])

    npTrainX = np.array([[2, 3], [5, 4], [6, 6]])
    npTrainY = np.array([1, 1, 2])

    knn = KNN(npTrainX, npTrainY)
    knn.find_nearest(np.array([4, 6]))


if __name__ == "__main__":
    main()
