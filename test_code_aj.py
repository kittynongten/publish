from nltk import NaiveBayesClassifier as nbc
import nltk
import re
import string
import pythainlp
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
import pickle

text = [input('\nข้อความ : ')]
print(text[0])

def clean_msg(msg):
    
    
    # ลบ text ที่อยู่ในวงเล็บ <> ทั้งหมด
    msg = re.sub(r'<.*?>','', msg)
    
    # ลบ hashtag
    msg = re.sub(r'#','',msg)
    
    # ลบ …
    msg = re.sub(r'…','',msg)
    
    # ลบ เครื่องหมายคำพูด (punctuation)
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c),'',msg)
    
    # ลบ separator เช่น \n \t
    msg = ' '.join(msg.split())
    
    return msg

clean_text = [clean_msg(txt) for txt in text]

print('original text:\n',text[0])
print('clean text:\n',clean_msg(text[0]))

nltk.download('words')
th_stop = tuple(thai_stopwords())
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()

def split_word(text):
            
    
    tokens = word_tokenize(text,engine='newmm')
    
    # Remove stop words ภาษาไทย และภาษาอังกฤษ
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop]
    #tokens = [i for i in tokens ]
    
    # หารากศัพท์ภาษาไทย และภาษาอังกฤษ
    # English
    tokens = [p_stemmer.stem(i) for i in tokens]
    

    # Thai
    tokens_temp=[]
    for i in tokens:
        w_syn = wordnet.synsets(i)
        if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
            tokens_temp.append(w_syn[0].lemma_names('tha')[0])
        else:
            tokens_temp.append(i)
    
    tokens = tokens_temp
    
    # ลบตัวเลข
    tokens = [i for i in tokens if not i.isnumeric()]
    
    # ลบช่องว่าง
    tokens = [i for i in tokens if not ' ' in i]

    return tokens

tokens_list = [split_word(txt) for txt in clean_text]
print('tokenized text:\n',split_word(clean_msg(text[0])))

tokens_list_j = [','.join(tkn) for tkn in tokens_list]
cvec = CountVectorizer(analyzer=lambda x:x.split(','))
c_feat = cvec.fit_transform(tokens_list_j)
print(c_feat)

vocab = cvec.vocabulary_
print(vocab)

c_feat[:,:20].todense()

tvec = TfidfVectorizer(analyzer=lambda x:x.split(','),)
t_feat = tvec.fit_transform(tokens_list_j)

t_feat[:,:5].todense()

print(len(tvec.idf_),len(tvec.vocabulary_))

c_feat[:,:5].todense()
print(c_feat.todense())

X = np.array(c_feat.todense())
#    test_input.append(0)
print(X)

new_model = pickle.load(open('nlp-model_new.pkl', 'rb'))

# Predict new observation's class
#y_predicted = new_model.predict(X)
y_predicted = new_model.predict([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

print(y_predicted)