#!/bin/python

from tensorflow.contrib import learn as lrn
import tensorflow as tf
import numpy as np


from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

import argparse
import sys

parser = argparse.ArgumentParser(description='get allele numbers table')

parser.add_argument('-s','--snps', action="store", dest='snps', type=str, help='012 file of SNPs for classifier training', nargs='?', default=None)
parser.add_argument('-t','--targets', action="store", dest='targets', type=str, help='target values', nargs='?', default=None)
parser.add_argument('-p','--testprop', action="store", dest='testprop', type=int, help='save every Nth value for testing', nargs='?', default=10)
parser.add_argument('-b','--batch', action="store_true", dest='batch', help='output tab delimited text', default=False)

args = parser.parse_args()


#digits = load_digits()
#X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

snps = np.genfromtxt(args.snps,dtype='int')
targets = np.genfromtxt(args.targets,dtype='str')

test = range(0,len(snps),args.testprop)
train = [I for I in range(0,len(snps)) if I not in test]

stst = snps[test,:]
ttst = targets[test,1]

strn = snps[train,:]
ttrn = targets[train,1]


#constant values (immutable)
classes = np.unique(targets[:,1])
n_classes = len(classes)
class2id=dict()
cids = list()
cid=0
print >>sys.stderr, "CLASSES: "
for C in classes:
    class2id[C]=cid
    print >>sys.stderr, "   ",cid,C
    cid +=1
    
ttrn = [class2id[C] for C in ttrn]
ttst = [class2id[C] for C in ttst]


#classifier = lrn.DNNClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=snps.shape[1])],
#                                                    hidden_units=[10, 20, 10],
#                                                    n_classes=n_classes)

classifier = lrn.DNNLinearCombinedClassifier(dnn_feature_columns=[tf.contrib.layers.real_valued_column("", dimension=snps.shape[1])],
                                                 dnn_hidden_units=[10,20,10],
                                                 n_classes=n_classes)
#classifier = lrn.DNNLinearCombinedClassifier(linear_feature_columns=[tf.contrib.layers.real_valued_column("", dimension=snps.shape[1])],
#                                                 #dnn_hidden_units=[10,20,10],
#                                                 n_classes=n_classes)
#classifier = lrn.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=snps.shape[1])], n_classes=n_classes)
classifier.fit(strn, ttrn, steps=50)

pred = classifier.predict(stst)

#setup counters
count=0
correct=0
classCount = dict()
classCorrect=dict()
for C in classes:
    classCount[C]=0
    classCorrect[C]=0
    
i=0
for P in pred:
    target = classes[ttst[i]]
    guess = classes[P]
    #print target, guess
    count+=1
    classCount[target]+=1
    if target==guess:
        classCorrect[target]+=1
        correct+=1
    i+=1

if args.batch:
    print round(correct/float(count),3),
    for C in classes:
        if classCount[C]>0:
            print "\t",round(classCorrect[C]/float(classCount[C]),2),
        else:
            print -1
else:
    print "TOTAL:",round(correct/float(count),3)
    for C in classes:
        if classCount[C]>0:
            print " ",C,":",round(classCorrect[C]/float(classCount[C]),2)
