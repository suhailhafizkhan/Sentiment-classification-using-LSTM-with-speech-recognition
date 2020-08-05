# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 01:52:16 2020

@author: Zoheb
"""
import re
import sqlite3
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import keras
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
import speech_recognition as sr
from tensorflow.keras.models import load_model


#database retrieval


conn = sqlite3.connect('database.sqlite')
data = pd.read_sql_query(''' SELECT * FROM Reviews WHERE Score !=3''',conn)


#assigning the target positives and negatives

score_num=list(data['Score'])
Polarity=[]
for i in score_num:
  if i > 3:
    Polarity.append(1)
  else:
    Polarity.append(0)


data['polarity']=Polarity
new_data=data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
new_data=new_data[new_data.HelpfulnessNumerator<=new_data.HelpfulnessDenominator]
yes=new_data[new_data['polarity']== 1]
no=new_data[new_data['polarity']== 0]
yess = yes.iloc[:50000]
noo = no.iloc[:50000]
df=yess.append(noo)
df = df.sample(frac=1).reset_index(drop=True)




def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase



preprocessed_reviews = []
for sentance in (df['Text'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    preprocessed_reviews.append(sentance.strip())
    
    
df['cleanedText'] = preprocessed_reviews
df['cleanedText'] = df['cleanedText'].astype('str') 


x=df['cleanedText']
y=df['polarity']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=355)



tokenize = Tokenizer(num_words=5000)
tokenize.fit_on_texts(x_train)

X_train_new = tokenize.texts_to_sequences(x_train)
X_test_new = tokenize.texts_to_sequences(x_test)






max_review_length = 600
X_train_new = sequence.pad_sequences(X_train_new, maxlen=max_review_length)
X_test_new = sequence.pad_sequences(X_test_new, maxlen=max_review_length)


#model building with 2 lstm layers

#commenting out the model because i have model loaded

'''
embed_vector_length = 32
model = Sequential()
model.add(Embedding(5000, embed_vector_length, input_length=max_review_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#monitor = EarlyStopping(monitor='val_loss',min_delta=0,mode='auto',restore_best_weights=True)
   
#model.fit(X_train_new, y_train, epochs=10,batch_size=128, verbose=1, validation_data=(X_test_new,y_test))
'''



#speech recognition and testing



mod=load_model(r'C:\Users\hp\Desktop\Amazon\senment.h5')
r = sr.Recognizer()


def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase




with sr.Microphone() as source:
  print("speak anything....")
  audio = r.listen(source,timeout=4)
  try:
    text = r.recognize_google(audio)
    print(type(text))
    text1=[text]
    print(type(text1))
    print(text1)
    preprocessed_reviews = []
    for sentance in (text1):
        sentance = re.sub(r"http\S+", "", sentance)
        sentance = decontracted(sentance)
        sentance = re.sub("\S*\d\S*", "", sentance).strip()
        sentance = re.sub('[^A-Za-z]+', ' ', sentance)
        preprocessed_reviews.append(sentance.strip())
    instance = tokenize.texts_to_sequences(preprocessed_reviews)
    flat_list=[]
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list= [flat_list]
    instance = sequence.pad_sequences(flat_list, maxlen=600)
    print("you said .. : ",text)
    result=mod.predict(instance)
    print(result)
    if result > 0.5:
        print("Positive")
    else:
        print("Negative")
  except:
    print("sorry could not hear....")
    

