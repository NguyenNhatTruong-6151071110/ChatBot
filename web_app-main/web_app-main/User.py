import streamlit as st
import tensorflow as tf

import pickle
import json
import random
# tf.compat.v1.enable_eager_execution()
from tkit import tokenize , stem_lower, encode
import numpy as np
# from tensorflow.keras.optimizers import SGD
from keras.models import load_model
 


model=load_model('web_app-main/model_train.h5')

import pickle
import json
with open('web_app-main/intents.json') as json_data:
    intents = json.load(json_data)
data = pickle.load(open("web_app-main/training_data", "rb"))
words_list = data['words']
tags_list = data['topics']
x_train = data['x_train']
y_train = data['y_train']


def predict_class(sentence, model):
    text=tokenize(sentence)
    text_encode = encode(text,words_list)
    res = model.predict(np.array([text_encode]))[0]

    
    results = [[i,r] for i,r in enumerate(res) if r>0.25]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": tags_list[r[0]], "probability": str(r[1])})
    return return_list
 
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
 
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

import streamlit as st
st.title('Welcome to Chatbot! ')
messeage=st.text_input('Write something....')
st.text_input(label="Client_chat", value=messeage)
st.text_input(label="Bot_chat", value=chatbot_response(messeage))



 