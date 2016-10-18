# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 22:57:04 2016

@author: yguo18

note ： Modify the directory of .csv and .dot
"""

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
from docutils.readers import Reader

allElectronicsData = open('trees.csv', 'r')
headers = allElectronicsData.readline()
reader = csv.reader(allElectronicsData)
headers = ['RID','age','income','student','credit','buy']
print(headers)
print(reader)
fearureList = []
labellist = []

for row in reader:
    labellist.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    fearureList.append(rowDict)

print(fearureList)

vec = DictVectorizer()
dummx = vec.fit_transform(fearureList).toarray()

print('dummx:' + str(dummx))
print(vec.get_feature_names())

print('labelList:' + str(labellist))

lb = preprocessing.LabelBinarizer()
dummy = lb.fit_transform(labellist)
print('dummy:' + str(dummy))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummx,dummy)
print('clf:' + str(clf))

with open('allElectronicInformation.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file = f)
oneRowx = dummx[0, :]
print('oneRowx:' + str(oneRowx))

newRowx = oneRowx

newRowx[0] = 1
newRowx[2] = 0
print('newRowx:' + str(newRowx))

predictedY = clf.predict(newRowx)
print('predictedY:' + str(predictedY))
