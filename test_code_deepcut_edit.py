# ใช้ตัดคำภาษาไทย
import deepcut
# ใช้งาน regex
import re
# จัดการเกี่ยวกับ array
import numpy as np
# สำหรับทำ classify และทดสอบโมเดล
from nltk import FreqDist, precision, recall, f_measure, NaiveBayesClassifier
from nltk.classify import apply_features
from nltk.classify import util
# สำหรับสร้างชุดข้อมูลสำหรับ train และ test เพื่อทดสอบประสิทธิภาพ
from sklearn.model_selection import KFold
import collections, itertools
from itertools import chain
import codecs
import os
import csv
import pickle

data_pos = [(line.strip(), 'pos') for line in open(".//data/pos.txt", 'r',encoding='utf8')]
data_neg = [(line.strip(), 'neg') for line in open(".//data/neg.txt", 'r',encoding='utf8')]


def split_words (sentence):
    return deepcut.tokenize(''.join(sentence.lower().split()))   

vocabulary = set(chain(*[deepcut.tokenize(i[0].lower()) for i in data_pos + data_neg]))
print("vocabulary : ",vocabulary)
sentences = [(split_words(sentence), sentiment) for (sentence, sentiment) in data_pos + data_neg]
print("sentences : ",sentences)


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in sentences:
        features['contains(%s)' % word] = (word in document_words)
    return features

features_data = np.array(sentences)
print("features_data : ",features_data)
# แบ่งข้อมูลเป็น 10 ชุด โดยไม่เรียง
k_fold = KFold(n_splits=10, random_state=1992, shuffle=True)
word_features = None
for train_set, test_set in k_fold.split(features_data):

    train_features = apply_features(extract_features, features_data[train_set].tolist())
    test_features = apply_features(extract_features, features_data[test_set].tolist())
    classifier = NaiveBayesClassifier.train(train_features)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test_features):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    
    accuracy_score = util.accuracy(classifier, test_features)
    print('train: {} test: {}'.format(len(train_set), len(test_set)))
    print('=================== Results ===================')
    print('Accuracy {:f}'.format(accuracy_score))
    print('            Positive     Negative')
    print('F1         [{:f}     {:f}]'.format(
        f_measure(refsets['pos'], testsets['pos']),
        f_measure(refsets['neg'], testsets['neg'])
    ))
    print('Precision  [{:f}     {:f}]'.format(
        precision(refsets['pos'], testsets['pos']),
        precision(refsets['neg'], testsets['neg'])
    ))
    print('Recall     [{:f}     {:f}]'.format(
        recall(refsets['pos'], testsets['pos']),
        recall(refsets['neg'], testsets['neg'])
    ))
    print('===============================================\n')

test_sentence = input('\nข้อความ : ')
test_word = {i:(i in deepcut.tokenize(test_sentence.lower())) for i in word_features}
print("test_word : ",test_word)
test_observed = classifier.classify(test_word)
print("test_observed ",test_observed)


