import codecs
import pandas as pd
from nltk import NaiveBayesClassifier as nbc # ใช้ในการเทรนข้อมูลน้อยๆหลักพัน หากเป็นหลักหมื่นไม่ควรใช้
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import multiprocessing
from gensim.models import Word2Vec
from numpy import array
from numpy import argmax
import numpy as np
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout, Activation
from tensorflow.keras.models import Sequential

texts = []
labels = []

#with codecs.open('./train.txt', 'r', "utf-8") as f: # โค้ดสำหรับเรียกไฟล์
#    lines = f.readlines()
#texts=[e.strip() for e in lines]
#f.close()

#with codecs.open("train_label.txt", 'r', "utf-8") as f: # โค้ดสำหรับเรียกไฟล์
#   lines = f.readlines()
#labels=[e.strip() for e in lines]
#f.close()

with open("train.txt",encoding="utf8") as f: # โค้ดสำหรับเรียกไฟล์
    for line in f:
        texts.append(line.strip())

with open("train_label copy.txt") as f: # โค้ดสำหรับเรียกไฟล์
    for line in f:
        labels.append(line.strip())

df = pd.DataFrame({"category": labels,"texts": texts})

df.to_csv("train_new.csv", index=False)
#print(traindata.shape)

neg_df = df[df.category == "neg"]
#print(neg_traindata.head())

pos_df = df[df.category == "pos"]
#print(pos_traindata.head())

#df["length"] = df["texts"].apply(word_tokenize).apply(len)
#print(traindata["length"].describe())
#print(traindata[traindata["length"]>=18].values.tolist())

#sentiment_df = pd.DataFrame({"category": labels,"texts": texts})
#Word2Vec
#cores = multiprocessing.cpu_count()
#print(cores)
#sentiment_df["words"] = df["texts"].apply(word_tokenize)
#print(sentiment_traindata.head())

#w2v_model = Word2Vec(sent, min_count=2, window=2,size=300,workers=cores-1)
#w2v_model['เสียดาย']

#print(sentiment_traindata['category'].sample(10))
#y_train = [row for row in sentiment_df['category']]

#Y_train = array(y_train)
#print(Y_train.shape)

#label_encoder = labelEncoder()
#integer_encoder = label_encoder.fit_transform(Y_train)
#print(integer_encoder)
#print(np.bicount(integer_encoder))

#onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoder = integer_encoder.reshape(len(integer_encoder), 1)
#print(integer_encoder.shape)

#onehot_encoder = onehot_encoder.fit_transform(integer_encoder)

#model = Sequential()
#model.add(Embedding(vocab_size+1, 32, input_length=max_length))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(3, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())

#Trian
#trian_model = model.fit(x_trian, y_train,
#        batch_size=32,
 #       epochs=20,
 #       verbose=1,
 #       validation_data=(x_valid, y_valid))

#print(plot_accuracy_and_loss(train_model))