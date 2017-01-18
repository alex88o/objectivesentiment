# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:45:15 2016

@author: alessandro
"""

import sqlite3 as lite
import sys

con = lite.connect('vocabulary.db')

with con:
    
    cur = con.cursor()    
    cur.execute("DROP TABLE IF EXISTS Words")
    cur.execute("CREATE TABLE Words(Id INTEGER PRIMARY KEY, \
							Word TEXT, \
							Tag TEXT, \
							PosScore FLOAT, \
							NegScore FLOAT, \
							ObjScore FLOAT)")
