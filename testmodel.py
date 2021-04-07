#from nltk import NaiveBayesClassifier as nbc # ใช้ในการเทรนข้อมูลน้อยๆหลักพัน หากเป็นหลักหมื่นไม่ควรใช้
from pythainlp.tokenize import word_tokenize
#import codecs
#from itertools import chain
#from sklearn.model_selection import train_test_split
#from sklearn import model_selection
import pickle
from sklearn.feature_extraction.text import CountVectorizer


#tokens_list_j = [','.join(tkn) for tkn in tokens_list]
cvec = CountVectorizer(analyzer=lambda x:x.split(','))

# load the model from disk
vocabulary = pickle.load(open('vocabulary.pkl', 'rb'))
loaded_model = pickle.load(open('nlp-model.pkl', 'rb'))
#c_feat = pickle.load(open('c_feat.pkl', 'rb'))
new_model = pickle.load(open('nlp-model_new.pkl', 'rb'))
y_predicted = pickle.load(open('y_predicted.pkl', 'rb'))

#print(c_feat)
#print(y_predicted)
#print(loaded_data)
print(vocabulary)

#test_sentence = input('\nข้อความ : ') # รับ input
#featurized_test_sentence = {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary} # แยกคำจากประโยคที่ถูก input เข้ามา
#print(new_model)
#print(featurized_test_sentence)
#print("test_sent:",test_sentence) # ปริ้น input
#print("tag:",loaded_model.classify(featurized_test_sentence)) # ใช้โมเดลที่ train ประมวลผลว่าเป็น tag ไหน
#y_predicted = model.predict(test_sentence)