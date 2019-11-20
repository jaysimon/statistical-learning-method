#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 Guiyang Duocaibao S&T Co.,Ltd. All rights reserved.
# FileName : perceptron.py_
# Author : Hou Wei
# Version : V1.0
# Date: 2019-10-18
# Description: Chapter 2 perceptron practical1
# History:

import numpy as np
import matplotlib.pyplot as plt


class Perceptron():
    def __init__(self, npWeight, npBias):
        self.npWeight = npWeight
        self.npBias = npBias
        self.fLR = 1.0
        # print("npWeight:\n", self.npWeight)
        # print("npBias:\n", self.npBias)

    def infer(self, npX):
        self.npY = np.matmul(npX, self.npWeight) + self.npBias
        pass

        return self.npY

    def train_ori(self, npX, npTag):
        """
        原始形式解法
        :param npX:
        :param npTag:
        :return:
        """
        self.npX = npX
        self.npTag = npTag
        npPred = self.infer(npX)
        npBinPred = np.zeros((npPred.shape))
        npBinPred[npPred > 0] = 1
        npBinPred[npPred < 0] = -1

        bDone = True
        while (True):
            bDone = True
            for iIndex in range(npX.shape[0]):
                npXi = npX[iIndex]
                npTagi = npTag[iIndex]

                if (self.infer(npXi) * npTagi <= 0):
                    bDone = False
                    self.npWeight = self.npWeight + self.fLR * npTagi * npXi
                    self.npBias = self.npBias + self.fLR * npTagi
                    # print("%d*X_1 + %d*X_2 + %d" %
                    #       (int(self.npWeight[0]), int(self.npWeight[1]),
                    #        int(self.npBias[0])))
                    self.draw()

            if (bDone):
                break
        # self.draw()
        print("final npWeight:\n", self.npWeight)
        print("final npBias:\n", self.npBias)

    def train_duality(self, npX, npTag):
        """
        对偶形式解法
        :param npX:
        :param npTag:
        :return:
        """
        self.npX = npX
        self.npTag = npTag
        npPred = self.infer(npX)
        npBinPred = np.zeros((npPred.shape))
        npBinPred[npPred > 0] = 1
        npBinPred[npPred < 0] = -1

        # 　Compute gram matrix
        npGram = np.zeros((npX.shape[0], npX.shape[0]))
        for i in range(npGram.shape[0]):
            for j in range(npGram.shape[1]):
                npGram[i, j] = np.sum(npX[i] * npX[j])
        # print(npGram)

        npAlpha = np.zeros(npX.shape[0])

        while (True):
            bDone = True
            for i in range(npX.shape[0]):
                fSum = np.sum(npAlpha * npTag.T * npGram[i])
                if (npTag[i] * (fSum + self.npBias) <= 0):
                    bDone = False
                    npAlpha[i] += 1
                    self.npBias = self.npBias + npTag[i]
                # print("npX(i):\n", npX[i], "\nnpY_i:\n", npTag[i])
                # print("fAlpha:\n", npAlpha, "\nfB:\n", self.npBias, "\nfSum:",
                #       npTag[i] * fSum + self.npBias, "\n")
            if (bDone):
                break
        print("final Alpha:\n", npAlpha, "\nBias:\n", self.npBias, "\n")

        for i in range(npX.shape[0]):
            self.npWeight = self.npWeight + npAlpha[i] * npX[i] * npTag[i]

        # self.draw()
        print("final Weight:\n", self.npWeight)
        print("final Bias:\n", self.npBias)

    def draw(self):
        for iIndex in range(self.npX.shape[0]):
            npXi = self.npX[iIndex]
            if (self.npTag[iIndex] > 0):
                plt.plot(npXi[0], npXi[1], "r*")
            else:
                plt.plot(npXi[0], npXi[1], "b*")
        x = np.arange(-10, 10)
        y = -(self.npBias + self.npWeight[0] * x) / self.npWeight[1]
        plt.xlim(np.min(self.npX[:, 0]) - 5, np.max(self.npX[:, 0]) + 5)
        plt.ylim(np.min(self.npX[:, 1]) - 5, np.max(self.npX[:, 1]) + 5)
        plt.title("%d*X_1 + %d*X_2 + %d" %
                  (int(self.npWeight[0]), int(self.npWeight[1]),
                   int(self.npBias[0])))
        plt.vlines(0, -10, 10, colors="r", linestyles="dashed")
        plt.hlines(0, -10, 10, colors="r", linestyles="dashed")
        plt.plot(x, y)
        plt.show()
        pass


def main():
    npWeight = np.array((0, 0))
    npBias = np.array((0))
    perceptron = Perceptron(npWeight, npBias)

    npX = np.array(([3, 3], [4, 3], [1, 1]))
    npY = np.array((1, 1, -1))
    npY = np.reshape(npY, (npY.shape[0], 1))

    # perceptron.train_ori(npX, npY)
    perceptron.train_duality(npX, npY)


if __name__ == "__main__":
    main()
