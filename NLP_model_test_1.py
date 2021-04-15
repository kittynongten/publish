from nltk import NaiveBayesClassifier as nbc # ใช้ในการเทรนข้อมูลน้อยๆหลักพัน หากเป็นหลักหมื่นไม่ควรใช้
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import pickle

with codecs.open('.//data/pos.txt', 'r', "utf-8") as f: # โค้ดสำหรับเรียกไฟล์
    lines = f.readlines()
listpos=[e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listpos
f.close()

with codecs.open('.//data/neg.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listneg=[e.strip() for e in lines] # ใช้ลบช่องว่างของตัวอักษรแล้วนำมาใส่ไว้ใน listneg
f.close()

pos1=['pos']*len(listpos) # ทำข้อมูล listpos ในให้เป็น pos
neg1=['neg']*len(listneg) # ทำข้อมูล listneg ให้เป็น neg
training_data = list(zip(listpos,pos1)) + list(zip(listneg,neg1)) # เทรนข้อมูล(คำ)ใน listpos และ listneg ให้เป็น neg และ pos แล้วเก็บไว้ใน training_data

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data])) # ทำการแบ่งคำโดยใช้ Pythainlp
print(vocabulary)
feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in training_data] # ทำการแบ่งคำในประโยคโดยใช้ Pythainlp
print(feature_set)
predictions = nbc.train(feature_set) # การเทรนโมเดล

test_sentence = input('\nข้อความ : ') # รับ input
featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary} # แยกคำจากประโยคที่ถูก input เข้ามา
print("test_sent : ",test_sentence) # ปริ้น input
print("featurized_test_sentence : ",featurized_test_sentence)
print("tag : ",predictions.classify(featurized_test_sentence)) # ใช้โมเดลที่ train ประมวลผลว่าเป็น tag ไหน

# save the data to disk
data = './/model/training_data.pkl'
pickle.dump(training_data, open(data, 'wb'))

# save the model to disk
model = './/model/nlp-model.pkl'
pickle.dump(predictions, open(model, 'wb'))

# save the vocabulary to disk
vocab = './/model/vocabulary.pkl'
pickle.dump(vocabulary, open(vocab, 'wb'))

#import pickle
model_new = './/model/nlp-model_new.pkl'
pickle.dump(model, open(model_new, 'wb'))

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)