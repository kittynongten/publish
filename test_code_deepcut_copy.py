# ใช้ตัดคำภาษาไทย
import deepcut
# ใช้งาน regex
import re
# จัดการเกี่ยวกับ array
import numpy as np
# สำหรับทำ classify และทดสอบโมเดล
from nltk import FreqDist, precision, recall, f_measure
from nltk.classify import apply_features
from nltk.classify import util
from nltk import NaiveBayesClassifier as nbc # ใช้ในการเทรนข้อมูลน้อยๆหลักพัน หากเป็นหลักหมื่นไม่ควรใช้
from pythainlp.tokenize import word_tokenize
from itertools import chain
# สำหรับสร้างชุดข้อมูลสำหรับ train และ test เพื่อทดสอบประสิทธิภาพ
from sklearn.model_selection import KFold
import collections, itertools
import codecs
# สำหรับบันทึกไฟล์ หรือ Model ไว้ใช้
import pickle

with codecs.open('pos.txt', 'r', "utf-8") as f: # โค้ดสำหรับเรียกไฟล์
    lines = f.readlines()
listpos=[e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listpos
f.close()

with codecs.open('neutral.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listneu=[e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listneu
f.close()

with codecs.open('neg.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listneg=[e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listneg
f.close()

pos1=['pos']*len(listpos) # ทำข้อมูล listpos ในให้เป็น pos
neu1=['neu']*len(listneg) # ทำข้อมูล listneg ให้เป็น neu
neg1=['neg']*len(listneg) # ทำข้อมูล listneg ให้เป็น neg
training_data = list(zip(listpos,pos1)) + list(zip(listneu,neu1)) + list(zip(listneg,neg1)) # เทรนข้อมูล(คำ)ใน listpos และ listneg ให้เป็น neg และ pos แล้วเก็บไว้ใน training_data

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data])) # ทำการแบ่งคำโดยใช้ Pythainlp
#print(vocabulary)
feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in training_data] # ทำการแบ่งคำในประโยคโดยใช้ Pythainlp
#print(feature_set)
predictions = nbc.train(feature_set) # การเทรนโมเดล

test_sentence = input('\nข้อความ : ') # รับ input
featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary} # แยกคำจากประโยคที่ถูก input เข้ามา
print("test_sent:",test_sentence) # ปริ้น input
print("tag:",predictions.classify(featurized_test_sentence)) # ใช้โมเดลที่ train ประมวลผลว่าเป็น tag ไหน