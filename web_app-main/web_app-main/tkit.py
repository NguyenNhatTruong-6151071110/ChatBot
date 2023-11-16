import nltk
from nltk import PorterStemmer
import numpy as np

nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem_lower(word):
    return stemmer.stem(word.lower())

# def encode(sentences, words_list):
#     data= [0 for i in range(len(words_list))]
#     words = [stem_lower(word) for word in sentences ]
#     for sent in words:
#         for inx, w in enumerate(words_list):
#             if w==sent:
#                 data[inx]=1
#     return np.array(data)

def encode(text, words_list, max_length=119):
    encoded_text = [0] * max_length
    for i, word in enumerate(text):
        if word in words_list and i < max_length:
            encoded_text[i] = words_list.index(word)
    return encoded_text



