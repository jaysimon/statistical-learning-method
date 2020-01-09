#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 Guiyang Duocaibao S&T Co.,Ltd. All rights reserved.
# FileName : decision-tree.py_
# Author : Hou Wei
# Version : V1.0
# Date: 2019-12-16
# Description: Decision tree code
# History:

import pandas as pd
import math


class DecisonTreeNode(object):
    def __init__(self, sFeature, sClass):
        """
        决策树节点
        :param sFeature: 当前节点特征
        :param sClass: 当前节点所属分类
        """
        self.sFeature = sFeature
        self.sClass = sClass
        self.dcNode = {}

    def add_leaf(self, sItemFeature, node):
        self.dcNode[sItemFeature] = node


class DecisionTree():
    def __init__(self, pdData, fMinInfoGain, sMethod):
        """
        初始化决策树
        :param fMinInfoGain: 最小信息增益/最小信息增益比
        :param sMethod: 分类算法: "ID3"(用信息增益进行分类) 或者　
                "C4.5"(用信息增益比进行分类)
        """
        self.pdData = pdData
        self.fMinInfoGain = fMinInfoGain
        self.sMethod = sMethod

    def entropy(self, pdData, sIndex):
        """
        计算指定pandas列的信息熵
        :param sIndex: 需要计算的索引
        :return:
        """
        lFeature = pdData.loc[:, sIndex].tolist()
        lSetFeature = list(set(lFeature))

        iLen = len(lFeature)
        fH = 0.0
        for sFeature in lSetFeature:
            iFreq = lFeature.count(sFeature)
            # print(sFeature, iFreq)
            fH -= (iFreq / iLen) * math.log((iFreq / iLen), 2)
        # print(fEntrop)
        return fH

    def conditional_entropy(self, pdData, sY, sX):
        """
        计算经验条件熵 H(Y|X)
        :param pdData:
        :param sY: 随机变量
        :param sX: 条件随机变量
        :return:
        """
        # print(pdData.loc[:, [sY, sX]])
        # print(pdData[pdData[sX] == "青年"])
        # print(self.entropy(pdData[pdData[sX] == "青年"], "类别"))

        lFeature = pdData.loc[:, sX].tolist()
        lSetFeature = list(set(lFeature))
        iLen = len(lFeature)
        fHYX = 0.0

        for sFeature in lSetFeature:
            iFreq = lFeature.count(sFeature)
            fHYX += (iFreq / iLen) * self.entropy(
                pdData[pdData[sX] == sFeature], sY)
        # print(fHYX)
        return fHYX

    def information_gain(self, pdData, sD, sA):
        """
        计算信息增益 g(D,A) = H(D) - H(D|A)
        :param pdData:
        :param sD:
        :param sA:
        :return:
        """
        fHD = self.entropy(pdData, sD)
        fHDA = self.conditional_entropy(pdData, sD, sA)
        fG = fHD - fHDA
        # print("fG:", fG)
        return fG

    def information_gain_ratio(self, pdData, sD, sA):
        """
        计算信息增益比 g_R(D,A) = g(D,A)/H(D)
        :param pdData:
        :param sD:
        :param sA:
        :return:
        """
        fHD = self.entropy(pdData, sD)
        fGDA = self.information_gain(pdData, sD, sA)
        fGR = fGDA / fHD
        return fGR

    def choose_feature(self, pdData, sD):
        """
        根据信息增益/信息增益比,选择最优特征
        :param pdData: pandas数据
        :param sD: 分类列名
        :return: 返回最优特征名
        """
        lIndex = pdData.columns.values.tolist()  # 获取特征列表
        # print(lIndex)
        lFeatures = lIndex[1:-1]
        # print(lFeatures)

        dcResult = {}

        if len(set(pdData.loc[:, sD].tolist())) == 1:
            return None, 0

        for sFeature in lFeatures:
            if (self.sMethod == "ID3"):
                dcResult[sFeature] = self.information_gain(pdData, sD, sFeature)
            elif (self.sMethod == "C4.5"):
                dcResult[sFeature] = self.information_gain_ratio(pdData, sD,
                                                                 sFeature)
            else:
                print("[Log] Wrong sMethod: ", self.sMethod)
        # print(dcResult)
        return max(dcResult, key=dcResult.get), dcResult[max(dcResult)]

    def get_max_class(self, pdData, sD):
        """
        返回数据中占最多数的类别
        :param pdData:
        :return:
        """
        lFeature = pdData.loc[:, sD].tolist()
        dcFeature = {}
        for item in lFeature:
            if item not in dcFeature.keys():
                dcFeature[item] = lFeature.count(item)
        return max(dcFeature, key=dcFeature.get)

    def create_node(self, pdData):
        sRootFeature, fMaxGain = self.choose_feature(pdData, "类别")

        # 返回条件,当类别只有一类,或者信息增益小于最小信息增益时
        if (len(set(pdData.loc[:, "类别"].tolist())) == 1 or
                fMaxGain < self.fMinInfoGain):
            node = DecisonTreeNode(sFeature=None,
                                   sClass=self.get_max_class(pdData, "类别"))
            return node

        sClass = self.get_max_class(self.pdData, "类别")
        node = DecisonTreeNode(sRootFeature, sClass)
        for sValue in set(pdData.loc[:, sRootFeature].tolist()):
            pdNewData = pdData[pdData[sRootFeature] == sValue].copy()
            pdNewData.drop(sRootFeature, axis=1, inplace=True)
            # print(pdNewData)
            node.add_leaf(sValue, self.create_node(pdNewData))

        return node


def main():
    sDataPath = "../data/贷款申请样本数据表-决策树.csv"

    pdData = pd.read_csv(sDataPath)

    # 分类算法: "ID3"(用信息增益进行分类)
    # 或者　
    # "C4.5"(用信息增益比进行分类)
    decisionTree = DecisionTree(pdData, fMinInfoGain=0.05, sMethod="C4.5")

    # print(decisionTree.choose_feature(pdData, "类别", "information_gain"))
    node = decisionTree.create_node(pdData)
    pass


if __name__ == "__main__":
    main()
