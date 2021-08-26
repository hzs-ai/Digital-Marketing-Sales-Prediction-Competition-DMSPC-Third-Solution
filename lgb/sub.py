#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 21:32:52 2021

@author: hzs
"""

import pandas as pd 

last_1 = pd.read_csv('../res/last_1.csv')

last_2 = pd.read_csv('../res/last_2.csv')

last_3 = pd.read_csv('../res/last_3.csv')


last_1['orders_3h_15h'] = (last_1['orders_3h_15h']+last_2['orders_3h_15h']+last_3['orders_3h_15h'])/3

last_1.to_csv('../res/last_all.csv',index=None)