import multiprocessing
from gensim.models import Word2Vec
import codecs
import pandas as pd
from nltk import NaiveBayesClassifier as nbc # ใช้ในการเทรนข้อมูลน้อยๆหลักพัน หากเป็นหลักหมื่นไม่ควรใช้
from pythainlp.tokenize import word_tokenize
import codecs
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cores = multiprocessing.cpu_count()

cores

sentiment_df["words"] = sentiment_df["text"].apply(word_tokenize)

sentiment_df.head()