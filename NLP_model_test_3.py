from nltk import FreqDist, precision, recall, f_measure, NaiveBayesClassifier
from nltk.classify import apply_features
from nltk.classify import util
from sklearn.model_selection import KFold
from pythainlp.tokenize import word_tokenize
from itertools import chain
import collections
import codecs
import pickle
import numpy

listpos  = []
listneg  = []
listneu  = []

with codecs.open('.//data/Dpositive.txt', 'r', "utf-8") as file: 
    line = file.readlines()
positive = [e.strip() for e in line]
for i in range(3):
    for word in positive : listpos.append(word)
file.close()

with codecs.open('.//data/Dnegative.txt', 'r', "utf-8") as file:
    line = file.readlines()
negative = [e.strip() for e in line]
for i in range(3):
    for word in negative: listneg.append(word)
file.close()

with codecs.open('.//data/Dneutral.txt', 'r', "utf-8") as file:
    line = file.readlines()
listneu  = [e.strip() for e in line]
file.close()

pos = ['pos']*len(listpos)
neg = ['neg']*len(listneg)
neu = ['neu']*len(listneg)

training_data = list(zip(listpos,pos)) + list(zip(listneg,neg))  + list(zip(listneu,neu))
print("\n\n\n#Step 1 : training_data\n\n\n ")
vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))
print("\n#Step 2 : vocabulary\n ")
feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},sentiment) for sentence, sentiment in training_data]
print("\n#Step 3 : feature_set\n ")
features_data = numpy.array(feature_set)
print("\n#Step 4 : features_data\n ")

print("\n#Step 5 : train & test\n ")
accuracy_score_total = 0
rounds = 10
number = 0
k_fold = KFold(n_splits=rounds, random_state=1992, shuffle=True)
for train_set, test_set in k_fold.split(features_data):
    train_features = features_data[train_set]
    test_features = features_data[test_set]

    prediction = NaiveBayesClassifier.train(train_features)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test_features):
        refsets[label].add(i)
        observed = prediction.classify(feats)
        testsets[observed].add(i)

    accuracy_score = util.accuracy(prediction, test_features)
    accuracy_score_total += accuracy_score
    number += 1
    print('No.',number,' >>> train: {} test: {}'.format(len(train_set), len(test_set)))
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

accuracy_score_average = accuracy_score_total/rounds
print(">> accuracy_score_average : ",accuracy_score_average)

test_sentence = input('\nข้อความ : ') 
while (test_sentence != 'exit'):
    featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary} 
    print("test_sentence : ",test_sentence) 
    print("word_tokenize : ",word_tokenize(test_sentence.lower()))
    print("sentiment : ",prediction.classify(featurized_test_sentence))  
    test_sentence = input('\nข้อความ : ') 
       