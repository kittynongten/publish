import numpy
# สำหรับทำ classify และทดสอบโมเดล
from nltk import FreqDist, precision, recall, f_measure, NaiveBayesClassifier
from nltk.classify import apply_features
from nltk.classify import util
# สำหรับสร้างชุดข้อมูลสำหรับ train และ test เพื่อทดสอบประสิทธิภาพ
from sklearn.model_selection import KFold
import collections, itertools
# สำหรับแบ่งคำ
import deepcut
from pythainlp.tokenize import word_tokenize
from itertools import chain
# สำหรับเปิด-ปิดไฟล์ และ export data
import codecs
import pickle

with codecs.open('.//data/pos.txt', 'r', "utf-8") as file: 
    lines = file.readlines()
listpos  =[e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listpos
file.close()

with codecs.open('.//data/neg.txt', 'r', "utf-8") as file:
    lines = file.readlines()
listneg = [e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listneg
file.close()

with codecs.open('.//data/neutral.txt', 'r', "utf-8") as file:
    lines = file.readlines()
listneu = [e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listneu
file.close()

pos = ['pos']*len(listpos) # ทำข้อมูล listpos ในให้เป็น pos
neg = ['neg']*len(listneg) # ทำข้อมูล listneg ให้เป็น neg
neu = ['neu']*len(listneg) # ทำข้อมูล listneu ให้เป็น neu

training_data = list(zip(listpos,pos)) + list(zip(listneg,neg)) + list(zip(listneu,neu)) # รวมข้อมูลใน listpos และ listneg มาเก็บไว้ใน training_data
print("\n\n\n#Step 1 : training_data\n\n\n ")
#print(training_data)
vocabulary = set(chain(*[deepcut.tokenize(i[0].lower()) for i in training_data]))
print("\n#Step 2 : vocabulary\n ")
#print("vocabulary : ",vocabulary)
feature_set = [({i:(i in deepcut.tokenize(sentence.lower())) for i in vocabulary},sentiment) for sentence, sentiment in training_data]
print("\n#Step 3 : feature_set\n ")
#print("feature_set : ",feature_set)
features_data = numpy.array(feature_set)
print("\n#Step 4 : features_data\n ")
#print("features_data : ",features_data)

accuracy_score_total = 0
rounds = 10
k_fold = KFold(n_splits=rounds, random_state=1992, shuffle=True)
for train_set, test_set in k_fold.split(features_data):
    train_features = features_data[train_set]
    test_features = features_data[test_set]
    print("\n#Step 5 : train_features + test_features\n ")
    #print("train_set : ",train_set)
    #print("test_set  : ",test_set)
    #print("train_features : ",train_features)
    #print("test_features  : ",test_features,"\n")

    prediction = NaiveBayesClassifier.train(train_features)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test_features):
        refsets[label].add(i)
        observed = prediction.classify(feats)
        testsets[observed].add(i)
        #print(i," observed  : ",observed, testsets[observed])
    print("\n#Step 6 : observed\n ")
    #print("refsets  : ",refsets)
    #print("testsets  : ",testsets)

    accuracy_score = util.accuracy(prediction, test_features)
    accuracy_score_total += accuracy_score
    print('train: {} test: {}'.format(len(train_set), len(test_set)))
    print('=================== Results ===================')
    print('Accuracy {:f}'.format(accuracy_score))
    for key in testsets.keys():
        print('F1         ',key,' :  {:f}'.format(f_measure(refsets[key], testsets[key])))
    for key in testsets.keys():
        print('Precision  ',key,' :  {:f}'.format(precision(refsets[key], testsets[key])))
    for key in testsets.keys():
        print('Recall     ',key,' :  {:f}'.format(recall(refsets[key], testsets[key])))

    print('===============================================\n')

accuracy_score_average = accuracy_score_total/rounds
print("accuracy_score_average : ",accuracy_score_average)

test_sentence = input('\nข้อความ : ') # รับ input
featurized_test_sentence =  {i:(i in deepcut.tokenize(test_sentence.lower())) for i in vocabulary} # แยกคำจากประโยคที่ถูก input เข้ามา
print("test_sent : ",test_sentence) # ปริ้น input
print("featurized_test_sentence : ",featurized_test_sentence)
print("sentiment : ",prediction.classify(featurized_test_sentence))   


