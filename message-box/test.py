# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 09:19:06 2018

@author: soumya
"""
#rm test.db

import os
import nltk
import sqlite3 as sq
from nltk.util import ngrams
from collections import Counter

corpus = []
path = './'
for root, dirs, i in os.walk(path):

    for file in i:
        if file.endswith('.txt'):
            f = open(os.path.join(path,file))
            corpus.append(f.read())
frequencies = Counter([])

for text in corpus:
    token = nltk.word_tokenize(text)
    bigrams = list(ngrams(token, 2))
    frequencies += Counter(bigrams)
    
conn = sq.connect('test.db')
print ("Opened database successfully")

conn.execute('''CREATE TABLE DATA
            (fwd TEXT,
                swd  TEXT,
                frequency INT NOT NULL);''')

itr = frequencies.most_common()
for elements in itr:
    i,j,v = elements[0][0] , elements[0][1], elements[1]
    data = (i,j,v)
    print(data)
    conn.execute("INSERT INTO DATA VALUES(?,?,?)", data)

conn.commit()
print ("Records created successfully");
conn.close()   
