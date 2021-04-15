import string
from string import punctuation
import nltk
from nltk.corpus import stopwords

def get_text_processing(text):
    stpword = stopwords.words('english')
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    return ' '.join([word for word in no_punctuation.split() if word.lower() not in stpword])

text = input('\nข้อความ : ')
print(text)
text_processing = get_text_processing(text)
print('original text   : ',text)
print('text_processing : ',text_processing)