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


class Node(object):
    def __init__(self, sFeature, sValue, sClass):
        """
        树节点
        :param sFeature: 当前分类特征
        :param sValue:  用于分类的特征值
        :param sClass:  当前节点所属类别
        """
        self.sFeature = sFeature
        self.sValue = sValue
        self.sClass = sClass
        self.left = None
        self.right = None


class CART():
    def __init__(self, sClass):
        """
        初始回归分类决策树
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
            node = Node(sFeature=None, sValue=None,
                        sClass=self.get_max_class(pdData))
            return node
        sRootIndex, sRootValue = self.choose_feature(pdData)

        sClass = self.get_max_class(pdData)
        node = Node(sRootIndex, sRootValue, sClass)

        # 生成左子树
        pdLeftData = pdData[pdData[sRootIndex] == sRootValue].copy()
        pdLeftData.drop(sRootIndex, axis=1, inplace=True)
        node.left = self.create_node(pdLeftData)

        # 生成右子树
        pdRightData = pdData[pdData[sRootIndex] != sRootValue].copy()
        pdRightData.drop(sRootIndex, axis=1, inplace=True)
        node.right = self.create_node(pdRightData)
        return node


def main():
    sDataPath = "../data/贷款申请样本数据表-决策树.csv"
    pdData = pd.read_csv(sDataPath)

    cart = CART("类别")
    node = cart.create_node(pdData)


if __name__ == "__main__":
    main()
