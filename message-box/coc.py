# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 22:54:56 2018

@author: soumya
"""

ip=open('science.txt','r')
#Edit should be in w mode
op=open('science1.txt','w')
for line in ip:
        line=line.strip().decode("ascii","ignore").encode("ascii")
        if line=="":continue
        op.write(line)
ip.close()
op.close()