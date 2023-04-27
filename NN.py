#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:39:53 2023

@author: mkr01
"""

from sklearn.datasets import load_iris
from pymfe.mfe import MFE

data = load_iris()

y = data.target
x = data.data

print(x)
print(y)

mfe = MFE(groups=["all"])

mfe.fit(x, y)

ft = mfe.extract()

print(ft)