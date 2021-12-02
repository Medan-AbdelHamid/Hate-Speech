# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 08:11:11 2019

@author: ASUS
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Reading the Data
#data_test = pd.read_csv("reviewsTs.csv")
data_test = pd.read_csv("cleantest500-2.csv")
y_test = data_test["ANOMALY"]
data_test.to_csv('text.csv', index = True )
testcleanWords = []
#testcleanWords=data_test["Full Text"]
print(y_test)
for i in range(data_test['FULLTEXT'].size):
    testcleanWords.append(data_test["FULLTEXT"][i])
    print(data_test["FULLTEXT"][i])
    print(i)
print("---Testestt Review Processing Done!---\n")
#tokenising Test data

# Load the tokenizer, so that we can use this tokenizer whenever we need to predict any reviews.
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
np.random.seed(1000)
num_most_freq_words_to_include = 5000
MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 500           # Input for keras.
embedding_vector_length = 32
#tokenizer = Tokenizer(num_words = num_most_freq_words_to_include)

test_review_tokenized = tokenizer.texts_to_sequences(testcleanWords)
#print(test_review_tokenized)
x_test = pad_sequences(test_review_tokenized, maxlen = MAX_REVIEW_LENGTH_FOR_KERAS_RNN) 
#print(x_test)              # 5000 X 500 
# Prediction.
# Load the model for future reference.

themodel = load_model('RNN LSTM modelNothing.h5')
#themodel = load_model('RNN GRU modelNothing.h5')
#history=themodel.fit(x_train, y_train, batch_size=64, epochs=15, validation_data=[x_val, y_val])
print('themodel ok')
ytest_prediction = themodel.predict(x_test)
print('prediction ok')
#print(ytest_prediction,y_test)
from sklearn.metrics import  roc_auc_score
print("The roc AUC socre model is : %.4f." %roc_auc_score(y_test, ytest_prediction)) 
# Creating csv file for 
# Changing the shape of ytest_prediction to 1-Dimensional

ytest_prediction = np.array(ytest_prediction).reshape((len(ytest_prediction), ))
for i in range(len(ytest_prediction)):
    ytest_prediction[i] = round(ytest_prediction[i]) 
ytest_prediction = ytest_prediction.astype(int)
'''
# Copy the predicted values to pandas dataframe with an id column, and a sentiment column.
output = pd.DataFrame(data = {"id": data_test["id"], "Psentiment": ytest_prediction, "Tsentiment": y_test})
if lstm == True:
    modelSelected = "LSTM"
else:
    modelSelected = "GRU"
fileName = "RNN " + modelSelected + " model" + method + ".h5"

outputName = "Predicted RNN " + modelSelected + " model" + method + ".csv"
output.to_csv(outputName, index = False, quoting = 3 )
'''
conf_matrix = confusion_matrix(y_test, ytest_prediction)

print(conf_matrix)
plt.figure(figsize=(4, 4))
#sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
plt.savefig("Confusion matrix.png", dpi=300, bbox_inches='tight')
probs = themodel.predict_proba(x_test)
#probs = probs[:, 1] 
auc = roc_auc_score(y_test, probs)  
print('AUC: %.2f' % auc) 

false_positive_r, true_positive_r, thresholds = metrics.roc_curve(y_test,probs)
roc_auc = metrics.auc(false_positive_r, true_positive_r)

#label = '{:.1f}% neg:{} pos:{} {}'.format(roc_auc * 100)

label = 'AUC  {:.3f}%'.format( roc_auc * 100)
plt.figure(1)
plt.plot(false_positive_r, true_positive_r, label=label)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC score(s)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', prop={'size': 10})
plt.savefig("ROC.png", dpi=300, bbox_inches='tight')
plt.grid()

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
from sklearn.metrics import auc
        # calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_test,probs)
        
auc_score = auc(recall, precision)
print('Auc_score %.3f   ' % auc_score)
        # plot the model precision-recall curve
label = 'AUC  {:.3f}%'.format( auc_score)
pyplot.figure(2)
pyplot.plot(recall, precision, marker='.', label=label)

        # axis labels
pyplot.title('precision_recall_curve')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
        # show the legend
pyplot.legend()
pyplot.savefig("AUC.png", dpi=300, bbox_inches='tight')

# Creating csv file for 
# Changing the shape of ytest_prediction to 1-Dimensional

ytest_prediction = np.array(ytest_prediction).reshape((len(ytest_prediction), ))
for i in range(len(ytest_prediction)):
    ytest_prediction[i] = round(ytest_prediction[i])
ytest_prediction = ytest_prediction.astype(int)

# Copy the predicted values to pandas dataframe with an id column, and a sentiment column.
#output = pd.DataFrame(data = {"id": data_test["id"], "Psentiment": ytest_prediction, "Tsentiment": y_test})
output = pd.DataFrame(data = { "Psentiment": ytest_prediction, "Tsentiment": y_test})

outputName = "Predicted RNN model"  + ".csv"
output.to_csv(outputName, index = False, quoting = 3 )


