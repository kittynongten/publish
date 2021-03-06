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
import codecs
import os
import csv

data_pos = [(line.strip(), 'pos') for line in open(".//word/pos.txt", 'r',encoding='utf8')]
data_neu = [(line.strip(), 'neu') for line in open(".//word/neutral.txt", 'r',encoding='utf8')]
data_neg = [(line.strip(), 'neg') for line in open(".//word/neg.txt", 'r',encoding='utf8')]

def split_words (sentence):
    return deepcut.tokenize(''.join(sentence.lower().split()))
sentences = [(split_words(sentence), sentiment) for (sentence, sentiment) in data_pos + data_neu + data_neg]
#(['word1', 'word2', 'word3', 'lastword'], 'label')
print(sentences)

def get_words_in_sentences(sentences):
    all_words = []
    for (words, sentiment) in sentences:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = FreqDist(wordlist)
    word_features = [word[0] for word in wordlist.most_common()]
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

features_data = np.array(sentences)
# แบ่งข้อมูลเป็น 10 ชุด โดยไม่เรียง
k_fold = KFold(n_splits=10, random_state=1992, shuffle=True)
word_features = None
accuracy_scores = []
for train_set, test_set in k_fold.split(features_data):
    word_features = get_word_features(get_words_in_sentences(features_data[train_set].tolist()))
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
    print('            Positive     neutral     Negative')
    print('F1         [{:f}     {:f}     {:f}]'.format(
        f_measure(refsets['pos'], testsets['pos']),
        f_measure(refsets['neu'], testsets['neu']),
        f_measure(refsets['neg'], testsets['neg'])
    ))
    print('Precision  [{:f}     {:f}     {:f}]'.format(
        precision(refsets['pos'], testsets['pos']),
        precision(refsets['neu'], testsets['neu']),
        precision(refsets['neg'], testsets['neg'])
    ))
    print('Recall     [{:f}     {:f}     {:f}]'.format(
        recall(refsets['pos'], testsets['pos']),
        precision(refsets['neu'], testsets['neu']),
        recall(refsets['neg'], testsets['neg'])
    ))
    print('===============================================\n')