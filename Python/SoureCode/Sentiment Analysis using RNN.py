# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:41:20 2018

@author: Mohammad Wasil Saleem.
"""

import re
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dropout, Conv1D, MaxPool1D, GRU, LSTM, Dense,Bidirectional
from keras.layers.core import Activation, RepeatVector, SpatialDropout1D
from keras.layers.core import Lambda
from tensorflow import expand_dims
import tensorflow as tf
from keras import regularizers
from keras.layers.wrappers import TimeDistributed
from gensim.models import Word2Vec,KeyedVectors

def training_Validation_Data(cleanWords, data_train):
    X = cleanWords
    y = data_train["ANOMALY"]
    
    test_start_index = int(data_train.shape[0] * .1)
    
    x_val = X[0:test_start_index]
    y_val = y[0:test_start_index]
    x_train = X[test_start_index:]
    y_train = y[test_start_index:]
    from sklearn.model_selection import train_test_split  
    x_train,x_val,y_train,y_val = train_test_split(X,y, test_size=0.02, random_state=0)


    return x_train, y_train, x_val, y_val

# Reading the Data
    
#data_train = pd.read_csv("reviewsTr.csv")
#data_test = pd.read_csv("reviewsTs.csv")    

data_train = pd.read_csv("Cleantrain2300NRU.csv")
data_test = pd.read_csv("Cleantest500-2.csv")    
#data_train = pd.read_csv("CleanL-HSAB-AbusHateTrain.csv")
#data_test = pd.read_csv("CleanL-HSAB-AbusHateTest.csv")
    
method = "Nothing"
# Input the value, whether you want to run the model on LSTM RNN or GRU .
print("Input 'LSTM' for LSTM RNN, 'GRU' for GRU RNN ")
modelInput= input("Do you want to compile the model using LSTM RNN or GRU RNN?\n")

if modelInput == "LSTM":
    lstm = True
else:
    lstm = False


# Let's process all the reviews together of train data.

cleanWords = []
for i in range(data_train['FULLTEXT'].size):
    if type(data_train["FULLTEXT"][i])==float:
        data_train["FULLTEXT"][i]='خطأ'
        print(i)
    cleanWords.append( data_train["FULLTEXT"][i])
#print(cleanWords)

print("---Review Processing Done!---\n")

# Splitting the data into tran and validation
x_train, y_train, x_val, y_val = training_Validation_Data(cleanWords, data_train)

# There is a data leakage in test set. 
#data_test["Anomaly"] = data_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = data_test["ANOMALY"]
#x_val=data_test["FULLTEXT"]
y_val=y_test
# Processing text dataset reviews.
testcleanWords = []
#testcleanWords=data_test["Full Text"]

for i in range(data_test['FULLTEXT'].size):
    if type(data_test["FULLTEXT"][i])==float:
        data_test["FULLTEXT"][i]='خطأ'
        print(i)
    testcleanWords.append(data_test["FULLTEXT"][i])
print("---Testestt Review Processing Done!---\n")
x_val=testcleanWords
# Generate the text sequence for RNN model
np.random.seed(1000)
num_most_freq_words_to_include = 5000
MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 500           # Input for keras.
embedding_vector_length = 300

all_review_list = x_train + x_val
tokenizer = Tokenizer(num_words = num_most_freq_words_to_include)
tokenizer.fit_on_texts(all_review_list)
#tokenisingtrain data
train_reviews_tokenized = tokenizer.texts_to_sequences(x_train)  
word_index = tokenizer.word_index
x_train = pad_sequences(train_reviews_tokenized, maxlen = MAX_REVIEW_LENGTH_FOR_KERAS_RNN)          # 20,000 x 500
#tokenising validation data
val_review_tokenized = tokenizer.texts_to_sequences(x_val)
x_val = pad_sequences(val_review_tokenized, maxlen = MAX_REVIEW_LENGTH_FOR_KERAS_RNN)               # 5000 X 500 

#tokenising Test data
test_review_tokenized = tokenizer.texts_to_sequences(testcleanWords)
x_test = pad_sequences(test_review_tokenized, maxlen = MAX_REVIEW_LENGTH_FOR_KERAS_RNN)               # 5000 X 500 

# Save the tokenizer, so that we can use this tokenizer whenever we need to predict any reviews.
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def RNNModel(lstm = False):
    MAX_SEQUENCE_LENGTH = 500
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300

    model = Sequential()
    
    model_emb = KeyedVectors.load('C:/Users/ASUS/Documents/Thesis/Code/Python/aravec-master/full_grams_sg_300_twitter/full_grams_sg_300_twitter.mdl')
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        #if word in model_emb:
        if model_emb.wv.__contains__(word): 
            #embedding_matrix[i] = model_emb[word]
            embedding_matrix[i] = model_emb.wv.__getitem__(word)
        else:
            embedding_matrix[i] = np.random.rand(1, EMBEDDING_DIM)[0]
   
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False))
    '''
    model.add(Embedding(input_dim = num_most_freq_words_to_include, 
                                output_dim = embedding_vector_length,
                                input_length=MAX_REVIEW_LENGTH_FOR_KERAS_RNN))
    '''
    model.add(Dropout(0.2))
    #model.add(Conv1D(filters = 128, kernel_size = 5, padding = 'same', activation = 'relu'))
    #model.add(Conv1D(filters = 128, kernel_size = 5, padding = 'same', activation = 'relu'))
    model.add(Conv1D(filters = 128, kernel_size = 5, padding = 'same', activation = 'relu'))
    model.add(MaxPool1D(pool_size = 2))
    if lstm == True:
        model.add(LSTM(100))
        #model.add(Bidirectional(LSTM(128)))
        
    else:
        model.add(GRU(100))
       
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.01)))             
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

def LSTMModel(lstm = False):
    MAX_SEQUENCE_LENGTH = 500
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300
    model = Sequential()
    '''
    model.add(Embedding(input_dim = num_most_freq_words_to_include, 
                                output_dim = embedding_vector_length,
                                input_length=MAX_REVIEW_LENGTH_FOR_KERAS_RNN))
    '''
    model_emb = KeyedVectors.load('C:/Users/ASUS/Documents/Thesis/Code/Python/aravec-master/full_grams_sg_300_twitter/full_grams_sg_300_twitter.mdl')
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        #if word in model_emb:
        if model_emb.wv.__contains__(word): 
            #embedding_matrix[i] = model_emb[word]
            embedding_matrix[i] = model_emb.wv.__getitem__(word)
        else:
            embedding_matrix[i] = np.random.rand(1, EMBEDDING_DIM)[0]

    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False))
   
    model.add(Bidirectional(LSTM(64)))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l1(0.01)))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model
s_vocabsize = num_most_freq_words_to_include
def GRUModel1():
    embedding_dim = 50
    vocab_size=5000
    maxlen=500
    model = Sequential()
    '''
    model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
    '''
    model.add(Embedding(input_dim = num_most_freq_words_to_include, 
                                output_dim = embedding_vector_length,
                                input_length=MAX_REVIEW_LENGTH_FOR_KERAS_RNN))

    #model.add(Bidirectional(LSTM(128)))
    #model.add(Dropout(0.2)) 
    #model.add(layers.Flatten())
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))    
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    return model
themodel = RNNModel(lstm)
#themodel =LSTMModel()

history=themodel.fit(x_train, y_train, batch_size=256, epochs=20, validation_data=[x_val, y_val])
themodel.summary()

# LSTM
#

# Creating file name for saving the model.
if lstm == True:
    modelSelected = "LSTM"
else:
    modelSelected = "GRU"
fileName = "RNN " + modelSelected + " model" + method + ".h5"

# Saving the model for future reference.
themodel.save(fileName)

# Prediction.
ytest_prediction = themodel.predict(x_test)

from sklearn.metrics import  roc_auc_score
print("The roc AUC socre is : %.4f." %roc_auc_score(y_test, ytest_prediction)) 


# Creating csv file for 
# Changing the shape of ytest_prediction to 1-Dimensional

ytest_prediction = np.array(ytest_prediction).reshape((len(ytest_prediction), ))
for i in range(len(ytest_prediction)):
    ytest_prediction[i] = round(ytest_prediction[i])
ytest_prediction = ytest_prediction.astype(int)

# Copy the predicted values to pandas dataframe with an id column, and a sentiment column.
#output = pd.DataFrame(data = {"id": data_test["id"], "Psentiment": ytest_prediction, "Tsentiment": y_test})
output = pd.DataFrame(data = { "Psentiment": ytest_prediction, "Tsentiment": y_test})

outputName = "Predicted RNN " + modelSelected + " model" + method + ".csv"
output.to_csv(outputName, index = False, quoting = 3 )

cm = confusion_matrix(y_test, ytest_prediction)

print(cm)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], '.-')
plt.plot(history.history['val_loss'], '.-')
plt.title('LOSS-EPOCH')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig("LOSS.png", dpi=300, bbox_inches='tight')
plt.grid()
'''
plt.plot(history.history['accuracy'], '.-')
plt.plot(history.history['val_accuracy'], '.-')
plt.title('ACC-EPOCH')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig("ACC.png", dpi=300, bbox_inches='tight')
plt.grid()
'''
loss, accuracy = themodel.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = themodel.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
