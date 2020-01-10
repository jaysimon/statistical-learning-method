#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 Houwei. All rights reserved.
# Project: 02-statistical-learning-method
# FileName : 05-02-CART.py
# Author : Hou Wei
# Version : V1.0
# Date: 2019/12/30 下午12:00
# Description: 05-02-CART
# History:

import pandas as pd
import math


class ClassficationNode(object):
    def __init__(self, sIndex, sValue, sClass):
        """
        分类树节点
        :param sIndex: 当前分类特征
        :param sValue:  用于分类的特征值
        :param sClass:  当前节点所属类别
        """
        self.sIndex = sIndex
        self.sValue = sValue
        self.sClass = sClass
        self.left = None
        self.right = None


class CT():
    def __init__(self, sClass):
        """
        初始分类决策树
        :param sClass: 用于分类的标签
        """
        self.sClass = sClass

    def gini(self, pdData):
        """
        计算基尼指数
        :param pdData:
        :return:
        """
        lFeature = pdData[self.sClass].tolist()
        setFeature = set(lFeature)

        fGini = 0
        iLen = len(lFeature)

        for sFeature in setFeature:
            iClassCount = lFeature.count(sFeature)
            fTmpGini = iClassCount * (iLen - iClassCount) / (iLen * iLen)
            fGini += fTmpGini
        return fGini

    def condition_gini(self, pdData, sIndex, sValue):
        """
        计算条件基尼指数
        :param pdData:
        :param sIndex:
        :param sValue:
        :return:
        """
        pdDataD1 = pdData[pdData[sIndex] == sValue]  # 取某值的样本集合
        pdDataD2 = pdData[pdData[sIndex] != sValue]  # 不取某值的样本集合

        fGiniD1 = (pdDataD1.shape[0] / pdData.shape[0]) * self.gini(pdDataD1)
        fGiniD2 = (pdDataD2.shape[0] / pdData.shape[0]) * self.gini(pdDataD2)
        return fGiniD1 + fGiniD2

    def get_max_class(self, pdData):
        """
        返回数据中占最多数的类别
        :param pdData:
        :return:
        """
        lFeature = pdData.loc[:, self.sClass].tolist()
        dcFeature = {}
        for item in lFeature:
            if item not in dcFeature.keys():
                dcFeature[item] = lFeature.count(item)
        return max(dcFeature, key=dcFeature.get)

    def choose_feature(self, pdData):
        """
        选择合适的分类特征
        :param pdData: 输入数据
        :return: 返回最佳分类特征, 最佳分类特征值
        """
        lIndex = pdData.columns.values.tolist()  # 获取特征列表
        lIndex.pop(0)  # 删除原来数据中的 "ID" 列
        lIndex.pop(-1)  # 删除原来数据中的 "类别" 列

        fMinGini = 100
        sBestIndex = ""
        sBestValue = ""
        for sIndex in lIndex:
            lValue = list(set(pdData.loc[:, sIndex].tolist()))  # 提取某个特征的所有可能值
            for sValue in lValue:
                fGini = self.condition_gini(pdData, sIndex, sValue)
                if (fGini < fMinGini):
                    fMinGini = fGini
                    sBestIndex = sIndex
                    sBestValue = sValue

        return sBestIndex, sBestValue

    def create_node(self, pdData):
        """
        递归创建节点
        :param pdData:
        :return:
        """
        if (len(set(pdData.loc[:, self.sClass].tolist())) == 1):
            node = ClassficationNode(sIndex=None, sValue=None,
                                     sClass=self.get_max_class(pdData))
            return node
        sRootIndex, sRootValue = self.choose_feature(pdData)

        sClass = self.get_max_class(pdData)
        node = ClassficationNode(sRootIndex, sRootValue, sClass)

        # 生成左子树
        pdLeftData = pdData[pdData[sRootIndex] == sRootValue].copy()
        pdLeftData.drop(sRootIndex, axis=1, inplace=True)
        node.left = self.create_node(pdLeftData)

        # 生成右子树
        pdRightData = pdData[pdData[sRootIndex] != sRootValue].copy()
        pdRightData.drop(sRootIndex, axis=1, inplace=True)
        node.right = self.create_node(pdRightData)
        return node


class RegressionNode(object):
    def __init__(self, sIndex, sSplitValue, fRegression):
        """
        回归树节点
        :param sIndex: 当前分类的特征
        :param sSplitValue: 分类特征点切分值
        :param fRegression: 回归值
        """
        self.sIndex = sIndex
        self.sSplitValue = sSplitValue
        self.fRegression = fRegression


class RT():
    def __init__(self, sRegression):
        """
        初始分类决策树
        :param sRegression: 用于分类的标签
        """
        self.sRegression = sRegression

    def choose_feature(self, pdData):
        """
        选择合适的分类特征
        :param pdData: 输入数据
        :return: 返回最佳分类特征, 最佳分类特征值
        """
        lIndex = pdData.columns.values.tolist()  # 获取特征列表
        # print(lIndex)
        lIndex.pop(-1)  # 去除掉Y标签,这里默认数据最后一列为标签.
        # print(lIndex)

        # 遍历计算平方误差,找到最小平方误差的变量及其切分点
        fMinSquareError = 10000000
        sMinIndex = ""
        fSplitValue = 0
        for sIndex in lIndex:
            lValue = pdData.loc[:, sIndex].tolist()
            for fValue in lValue[0:-1]:
                fSquareError = self.square_error(pdData, sIndex, fValue)
                # print(fSquareError)
                if (fSquareError < fMinSquareError):
                    fMinSquareError = fSquareError
                    sMinIndex = sIndex
                    fSplitValue = fValue
        # print(fMinSquareError, sMinIndex, fSplitValue)
        return fMinSquareError, sMinIndex, fSplitValue

    def square_error(self, pdData, sIndex, sSplitValue):
        """
        计算平方误差
        :param pdData: 输入数据
        :param sIndex: 进行计算的特征
        :param sSplitValue: 用于切分的特征值
        :return:
        """
        pdDataR1 = pdData[pdData[sIndex] <= sSplitValue]
        pdDataR2 = pdData[pdData[sIndex] > sSplitValue]

        fAverYR1 = self.__aver__Y(pdDataR1)
        fAverYR2 = self.__aver__Y(pdDataR2)

        lValueR1 = pdDataR1.loc[:, sIndex].tolist()
        lValueR2 = pdDataR2.loc[:, sIndex].tolist()

        fMeanR1 = 0
        fMeanR2 = 0
        for fValueR1 in lValueR1:
            fMeanR1 += math.pow((fValueR1 - fAverYR1), 2)
        for fValueR2 in lValueR2:
            fMeanR2 += math.pow((fValueR2 - fAverYR2), 2)

        return fMeanR1 + fMeanR2

    def __aver__Y(self, pdData):
        """
        计算数据内的标记平均值
        :param pdData:
        :return:
        """
        lRegressionValue = pdData.loc[:, self.sRegression].tolist()

        fResult = 0
        for fRegressionValue in lRegressionValue:
            fResult += fRegressionValue
        return fResult / len(lRegressionValue)

    def create_node(self, pdData):
        """
        递归创建节点
        :param pdData:
        :return:
        """

        # print(pdData, "\n")
        self.choose_feature(pdData)

        if (len(set(pdData.loc[:, self.sRegression].tolist())) == 1):
            fAverRegression = self.__aver__Y(pdData)
            node = RegressionNode(sIndex=None, sSplitValue=None,
                                  fRegression=fAverRegression)
            return node
        fMinSquareError, sRootIndex, sRootSplitValue = self.choose_feature(
            pdData)
        fAverRegression = self.__aver__Y(pdData)
        node = RegressionNode(sRootIndex, sRootSplitValue, fAverRegression)

        # 生成左子树
        print(pdData, "\n")
        pdLeftData = pdData[pdData[sRootIndex] <= sRootSplitValue].copy()
        # pdLeftData.drop(sRootIndex, axis=1, inplace=True)
        node.left = self.create_node(pdLeftData)

        # 生成右子树
        pdRightData = pdData[pdData[sRootIndex] > sRootSplitValue].copy()
        # pdRightData.drop(sRootIndex, axis=1, inplace=True)
        node.right = self.create_node(pdRightData)
        return node


def main():
    # 分类树demo

    # sDataPath = "../data/贷款申请样本数据表-决策树.csv"
    # pdData = pd.read_csv(sDataPath)
    #
    # cart = CT("类别")
    # node = cart.create_node(pdData)

    # 回归树demo
    sDataPath = "../data/regression-tree-data.csv"
    pdData = pd.read_csv(sDataPath)

    # print(pdData[pdData["x"] <= 5])
    rt = RT("y")
    node = rt.create_node(pdData)
    pass


if __name__ == "__main__":
    main()
