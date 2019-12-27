#!/usr/bin/env python
# -*- coding: utf-8 -*- #
# Copyright (C) 2018 Guiyang Duocaibao S&T Co.,Ltd. All rights reserved.
# Project: 02-statistical-learning-method
# FileName : 05-1-pickle-save-class.py
# Author : Hou Wei
# Version : V1.0
# Date: 2019/12/26 上午11:07
# Description: 05-pickle-test
# History:

import pickle


class Test():
    def __init__(self):
        pass

    def infer(self, sInput):
        print(sInput)


def main():
    test = Test()

    with open("../data/test.pk", "wb") as f:
        pickle.dump(test, f)

    with open("../data/test.pk", "rb") as f:
        stest = pickle.load(f)
    stest.infer("dsfa")


if __name__ == "__main__":
    main()
