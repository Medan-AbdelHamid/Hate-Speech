from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

import gensim
from gensim import corpora, models
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors

from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

#from nltk.corpus import reuters 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, NuSVC,SVC
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc,average_precision_score
from sklearn.model_selection import KFold,LeaveOneOut,ShuffleSplit,StratifiedKFold
from sklearn.model_selection import cross_val_score 	
from sklearn.pipeline import make_pipeline

import numpy as np
import pickle
import timeit
import time
import matplotlib.pyplot as plt
from matplotlib import pyplot
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import(SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,KMeansSMOTE)
from collections import Counter

#stop_words = set(stopwords.words('english'))
p_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('arabic'))
def word_transform(doc_set, ldamodel, dictionary,no_topics):
	data = []
	# loop through document list
	# ('1')
	for i in doc_set:
		#input('01')
		doc = ' '.join([word for word in i.split() if word not in  stop_words])
		doc = doc.lower()
		doc = doc.split()
		#print(doc)
		#input('02doc')
		doc_prob = [0] * no_topics
		for j in doc:
			if j in stop_words:
				continue
			word_dist = ldamodel.get_document_topics(dictionary.doc2bow(list(j)))#ldamodel[dictionary.doc2bow(list(j))]
			#print(j,word_dist)
			for k in word_dist: 
				doc_prob[int(k[0])] = doc_prob[int(k[0])] + float(k[1])
		data.append(doc_prob )
		#print(doc_prob)
		#input('03doc_prob')
	return data

def topic_transform(doc_set, ldamodel, dictionary,no_topics):
	data = []
	# loop through document list
	for i in doc_set:
		doc = ' '.join([word for word in i.split() if word not in stop_words])
		doc = doc.lower()
		#print(doc)
		doc_lda = ldamodel[dictionary.doc2bow(tokenizer.tokenize(doc))]
		#print(dictionary.doc2bow(tokenizer.tokenize(doc)))
		doc_prob = [0] * no_topics
		#print(doc_lda)
		for j in doc_lda:
			#print(float(j[1]))
			#print(int(j[0]))
			#input('03')
			doc_prob[int(j[0])] = float(j[1])
		data.append(doc_prob)
	return data
def tokenize(doc_set):
	# list for tokenized documents in loop
	texts = []
	tokenizer = RegexpTokenizer(r'\w+')
	
	# loop through document list
	for i in doc_set:
		try:
			# clean and tokenize document string
			#raw = i.lower()
			tokens = tokenizer.tokenize(i)

			# remove stop words from tokens
			stopped_tokens = [i for i in tokens if not i in stop_words]

			# stem tokens
			stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
			# add tokens to list
			#print(stopped_tokens)
			texts.append(stopped_tokens)
		except:
			print(i)
	return texts


def create_corpus(doc_set):
	texts = tokenize(doc_set)
	#print(texts)
	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(texts)
	dictionary.filter_extremes(no_below=1, no_above=0.5)
	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in texts]
	return dictionary, corpus

def load_glove_data():
	# read the word to vec file
	GLOVE_6B_100D_PATH = "glove.6B.100d.txt"
	dim = 100
	glove_small = {}
	with open(GLOVE_6B_100D_PATH, "rb") as infile:
		for line in infile:
			parts = line.split()
			try:
				word = parts[0].decode("utf-8")
				x = []
				for i in range(len(parts)-1):
					x.append(float(parts[i+1].decode("utf-8")))
				glove_small[word] = x
			except: 
				print('')
	return glove_small

def word2vec_transform(dataset, word2vec, dim):
	trans_data = []
	for doc in dataset:
		words = doc.lower().split()
		#words = doc.split()
		w_length = 1
		data = np.zeros(dim)
		for i in range(len(words)):
			#if words[i] in word2vec and words[i] not in stop_words:
			if word2vec.wv.__contains__(words[i]) and words[i] not in stop_words:    
				data = data + word2vec.wv.__getitem__(words[i])#[words[i]]
				w_length = w_length + 1
		data = data / float(w_length)
		trans_data.append(data)
	return trans_data

def load_data(path):
	X, y = [], [] 
	with open(path, "r") as infile:
		for line in infile:
			label, text = line.split("\t")
        	# texts are already tokenized, just split on space
        	# in a real case we would use e.g. spaCy for tokenization
        	# and maybe remove stopwords etc.
			X.append(text)
			y.append(label)
	return X, np.array(y)
def load_csv_data(path):
    dataset = pd.read_csv(path)
    X, y = [], []
    text = dataset['FULLTEXT']  
    label = dataset["ANOMALY"]
    i=0
    for line in text:
        X.append(text[i])
        i=i+1
    i=0    
    for line in label:
        y.append(label[i])
        i=i+1
    return X, np.array(y)
def tf_classify(train_docs, train_labels, test_docs, test_labels):
    
	text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))), ('clf',KNeighborsClassifier(n_neighbors=1)) ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	
	confusion_matrix = metrics.confusion_matrix(test_labels,predicted)
	#print(confusion_matrix)
	f1_score = metrics.f1_score(test_labels,predicted)
	#print("tf_KNN    " + "acc =",str(100 * np.mean(predicted == test_labels))," F1 =",str(100*f1_score ))
	# prediction
	negative = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 0])
	positive = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 1])

	#plot_auc(test_docs, test_labels, text_clf, "KNeighborsClassifier_tf", negative, positive)
	text_clf = Pipeline([('vect', CountVectorizer()), ('clf', SGDClassifier(random_state=42)), ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	
	confusion_matrix = metrics.confusion_matrix(test_labels,predicted)
	f1_score = metrics.f1_score(test_labels,predicted)
	#print(confusion_matrix)
	print("tf_SVM    " + "acc =",str(100 * np.mean(predicted == test_labels))," F1 =",str(100*f1_score ))
	negative = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 0])
	positive = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 1])

	#plot_auc(test_docs, test_labels, text_clf, "SGDClassifier_tf", negative, positive)
	'''
	text_clf = Pipeline([('vect', CountVectorizer()), ('clf', XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)), ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	
	confusion_matrix = metrics.confusion_matrix(test_labels,predicted)
	print(confusion_matrix)
	print("tf_XGB    " + str(100 * np.mean(predicted == test_labels))  )
	negative = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 0])
	positive = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 1])

	plot_auc(test_docs, test_labels, text_clf, "XGBClassifier_tf", negative, positive)
	'''
def tf_idf_classify(train_docs, train_labels, test_docs, test_labels):
	text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',KNeighborsClassifier(n_neighbors=1)) ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	output = pd.DataFrame(data = { "Psentiment":predicted, "Tsentiment": test_labels})

	fileName = "predtf_idf_knn"  + ".csv"

	output.to_csv(fileName, index = False, quoting = 3 )

	
	confusion_matrix = metrics.confusion_matrix(test_labels,predicted)
	#print(confusion_matrix)
	f1_score = metrics.f1_score(test_labels,predicted)
	#print("tf_idf_KNN    " +"acc =",str(100 * np.mean(predicted == test_labels))," F1 =",str(100*f1_score )) 
	negative = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 0])
	positive = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 1])

	#plot_auc(test_docs, test_labels, text_clf, "KNeighborsClassifier_tfidf", negative, positive)

	text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(random_state=42)), ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	confusion_matrix = metrics.confusion_matrix(test_labels,predicted)
	#print(confusion_matrix)
	f1_score = metrics.f1_score(test_labels,predicted)
	print("tf_idf_SVM    " + "acc =",str(100 * np.mean(predicted == test_labels))," F1 =",str(100*f1_score ))   
	negative = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 0])
	positive = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 1])

	#plot_auc(test_docs, test_labels, text_clf, "SGDClassifier_tfidf", negative, positive)
	'''
	text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)), ])
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	confusion_matrix = metrics.confusion_matrix(test_labels,predicted)
	print(confusion_matrix)
	print("tf_idf_XGB    " + str(100 * np.mean(predicted == test_labels))  )
	negative = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 0])
	positive = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 1])

	plot_auc(test_docs, test_labels, text_clf, "XGBClassifier_tfidf", negative, positive)

def word2vec_classify(train_docs, train_labels, test_docs, test_labels):
	word2vec = load_glove_data()
	train_docs = word2vec_transform(train_docs, word2vec, 100)
	test_docs = word2vec_transform(test_docs, word2vec, 100)
	text_clf = KNeighborsClassifier(n_neighbors=1)
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	print("word2vec_KNN    " + str(100 * np.mean(predicted == test_labels)) )
	svm = SGDClassifier(random_state=420, tol=.001)#MultinomialNB()
	svm.fit(train_docs, train_labels)  
	predicted = svm.predict(test_docs)
	print("word2vec_SVM    " + str(100 * np.mean(predicted == test_labels))  )
'''

def word2vec_classify(train_docs, train_labels, test_docs, test_labels,val_docs, val_labels):
	word2vec, dimension = load_vectors()
	train_docs = word2vec_transform(train_docs, word2vec,dimension)
	test_docs = word2vec_transform(test_docs, word2vec,dimension)
	val_docs = word2vec_transform(val_docs, word2vec,dimension)	
	#scaler = preprocessing.StandardScaler().fit(train_docs)
	#train_docs = scaler.transform(train_docs)
	#test_docs = scaler.transform(test_docs)
	#val_docs = scaler.transform(val_docs)
	#train_docs = preprocessing.scale(train_docs)
	#test_docs = preprocessing.scale(test_docs)
	#val_docs = preprocessing.scale(val_docs)
	print('Resampled train shape %s' % Counter(train_labels))
	print('Resampled test shape %s' % Counter(test_labels))
	print('Resampled test shape %s' % Counter(val_labels))
	#text_clf = RandomForestClassifier(n_estimators=300)#SGDClassifier(random_state=420)#MultinomialNB()
	text_clf =CatBoostClassifier(iterations=250,learning_rate=0.3,depth=5,loss_function='Logloss',border_count=32,random_state=420)# ,task_type="GPU")
	text_clf.fit(train_docs, train_labels)
	predictedtest = text_clf.predict(test_docs)
	predictedval = text_clf.predict(val_docs)
	classifier_name = text_clf.__class__.__name__
	print('classifier_name',classifier_name)
	'''
	output = pd.DataFrame(data = { "Psentiment":predicted, "Tsentiment": test_labels})

	fileName = "predword2vec_SGD"  + ".csv"

	output.to_csv(fileName, index = False, quoting = 3 )
	'''
	confusion_matrix_test = metrics.confusion_matrix(test_labels,predictedtest)
	confusion_matrix_val = metrics.confusion_matrix(val_labels,predictedval)
	#print(confusion_matrix)
	#f1_score = metrics.f1_score(test_labels,predicted)
	#print("word2vec_SVM    " + "acc =",str(100 * np.mean(predicted == test_labels))," F1 =",str(100*f1_score ))  
	negativeval = len(text_clf.predict(val_docs)[text_clf.predict(val_docs) == 0])
	positiveval = len(text_clf.predict(val_docs)[text_clf.predict(val_docs) == 1])

	negativetest = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 0])
	positivetest = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 1])

	plot_auc(test_docs, test_labels, text_clf, classifier_name+"_word2vec_test", negativetest, positivetest,1)
	plot_auc(val_docs, val_labels, text_clf, classifier_name+"_word2vec_val", negativeval, positiveval,2) 

	plot_roc(test_docs, test_labels, text_clf, classifier_name+"_word2vec_test", negativetest, positivetest,3)
	plot_roc(val_docs, val_labels, text_clf,classifier_name+ "_word2vec_val", negativeval, positiveval,4) 
	
	average_precision = average_precision_score(test_labels,predictedtest)

	print('Average precision-recall score: {0:0.2f}'.format(average_precision*100))
	print( text_clf.get_params())
	#print('iteration',text_clf.get_best_iteration())

def word2vec_classifyGridSearchCV(train_docs, train_labels, test_docs, test_labels):

	word2vec, dimension = load_vectors()
	train_docs = word2vec_transform(train_docs, word2vec,dimension)
	test_docs = word2vec_transform(test_docs, word2vec,dimension)
	
	#pipeline = make_pipeline(preprocessing.StandardScaler(),CatBoostClassifier(depth=6,border_count=32,random_state=420,loss_function='Logloss' ,task_type="GPU"))
	estemater=CatBoostClassifier(depth=5,border_count=32,random_state=420,loss_function='Logloss' ,task_type="GPU")
	#estemater = XGBClassifier()#objective= 'binary:logistic',nthread=4,seed=42)
	#hyperparameters = { 'catboostclassifier__iterations' : [250,100,500],'catboostclassifier__learning_rate':[0.03,0.001,0.01,0.1,0.3]}
	hyperparameters = { 'iterations' : [250,100,500],'learning_rate':[0.03,0.001,0.01,0.1,0.05]}
	#hyperparameters = {'n_estimators': range(60, 220, 40) ,'learning_rate': [0.1, 0.01, 0.05]}
	skf= ShuffleSplit(n_splits=5,random_state=100)
	scorers = {
			#'f1_score': metrics.make_scorer(metrics.f1_score),
			'roc_auc_score':  metrics.make_scorer(metrics.roc_auc_score)
			}

	#text_clf = GridSearchCV(pipeline, hyperparameters, cv=skf.split(train_docs, train_labels),scoring = scorers, refit = "roc_auc_score")
	#text_clf = GridSearchCV(estimator=estemater,param_grid=hyperparameters, cv=skf.split(train_docs, train_labels),scoring = scorers, refit = "roc_auc_score")
	text_clf = GridSearchCV(estimator=estemater,param_grid=hyperparameters,scoring ='roc_auc',cv = 5,refit = "roc_auc",return_train_score="True")
	print('Resampled train shape %s' % Counter(train_labels))
	print('Resampled test shape %s' % Counter(test_labels))

	text_clf.fit(train_docs, train_labels)
	predictedtest = text_clf.predict(test_docs)
	classifier_name = text_clf.__class__.__name__
	print('classifier_name',classifier_name)
	#best_model = text_clf.best_estimator_
	negativetest = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 0])
	positivetest = len(text_clf.predict(test_docs)[text_clf.predict(test_docs) == 1])

	plot_auc(test_docs, test_labels, text_clf, classifier_name+"_word2vec_test", negativetest, positivetest,1)
	plot_roc(test_docs, test_labels, text_clf, classifier_name+"_word2vec_test", negativetest, positivetest,3)
	best_model = text_clf.best_estimator_
	pred = best_model.predict(test_docs)
	negativetest = len(best_model.predict(test_docs)[best_model.predict(test_docs) == 0])
	positivetest = len(best_model.predict(test_docs)[best_model.predict(test_docs) == 1])
	plot_auc(test_docs, test_labels, best_model, classifier_name+"_word2vec_testP", negativetest, positivetest,4)
	plot_roc(test_docs, test_labels, best_model, classifier_name+"_word2vec_testP", negativetest, positivetest,5)

	
	average_precision = average_precision_score(test_labels,predictedtest)

	print('Average precision-recall score: {0:0.2f}'.format(average_precision*100))
	print( text_clf.get_params())
	# Dictionary of best parameters
	print('best_pars',text_clf.best_params_)
	# Best catboostclassifier model that was found based on the metric score you specify
	best_model = text_clf.best_estimator_
	# Save model
	pickle.dump(text_clf.best_estimator_, open("catboost.pickle", "wb"))
	dbfile = open('catboost.pickle', 'rb')      
	db = pickle.load(dbfile) 
	print('params',db.get_params())
	predictedtest = db.predict(test_docs)
	print('f1_score',metrics.f1_score(predictedtest,test_labels)*100)
	input('04best_ params')

def LDA_classify(train_docs, train_labels, test_docs, test_labels):
	text_clf = KNeighborsClassifier(n_neighbors=1)
	text_clf.fit(train_docs, train_labels) 
	predicted = text_clf.predict(test_docs)
	print('Resampled train shape %s' % Counter(train_labels))
	print('Resampled test shape %s' % Counter(test_labels))

	output = pd.DataFrame(data = { "Psentiment":predicted, "Tsentiment": test_labels})

	fileName = "LDA_KNN"  + ".csv"

	output.to_csv(fileName, index = False, quoting = 3 )
	KNN_accu = 100 * np.mean(predicted == test_labels)
	confusion_matrix = metrics.confusion_matrix(test_labels,predicted)
	#print("KNeighborsClassifier_LDA")
	#print(confusion_matrix)
	f1_score = metrics.f1_score(test_labels,predicted)
	#print("LDA_KNN    " + "acc =",str(100 * np.mean(predicted == test_labels))," F1 =",str(100*f1_score ))  

	#svm = SGDClassifier(random_state=42)#MultinomialNB()
	svm = GaussianNB()#MultinomialNB()
	svm.fit(train_docs, train_labels)  
	predicted = svm.predict(test_docs)
	confusion_matrix = metrics.confusion_matrix(test_labels,predicted)
	print("SGDClassifier_LDA")
	print(confusion_matrix)
	f1_score = metrics.f1_score(test_labels,predicted)
	print("LDA_SVM    " + "acc =",str(100 * np.mean(predicted == test_labels))," F1 =",str(100*f1_score ))  

	SVM_accu = 100 * np.mean(predicted == test_labels)
	rf = RandomForestClassifier(n_estimators=100)#MultinomialNB()
	rf.fit(train_docs, train_labels)  
	predicted = rf.predict(test_docs)
	#print("RandomForestClassifier_LDA")
	confusion_matrix = metrics.confusion_matrix(test_labels,predicted)
	#print(confusion_matrix)
	RF_accu = 100 * np.mean(predicted == test_labels)

	return KNN_accu,SVM_accu,RF_accu
def plot_auc(test_docs, test_labels, estimator, estimator_name, neg, pos,k):
        try:
            classifier_probas = estimator.decision_function(test_docs)
        except AttributeError:
            classifier_probas = estimator.predict_proba(test_docs)[:, 1]

        false_positive_r, true_positive_r, thresholds = metrics.roc_curve(test_labels, classifier_probas)
        roc_auc = metrics.auc(false_positive_r, true_positive_r)
                
        label = '{:.1f}% neg:{} pos:{} {}'.format(roc_auc * 100, neg, pos, estimator_name)
        plt.figure(k)
        plt.plot(false_positive_r, true_positive_r, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('ROC score(s)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right', prop={'size': 10})
        plt.savefig(estimator_name+".png", dpi=300, bbox_inches='tight')
        plt.grid()
'''
def plot_auc( estimator, estimator_name, neg, pos):
        try:
            classifier_probas = estimator.decision_function(X_test)
        except AttributeError:
            classifier_probas = estimator.predict_proba(X_test)[:, 1]

        false_positive_r, true_positive_r, thresholds = metrics.roc_curve(self.y_test, classifier_probas)
        roc_auc = metrics.auc(false_positive_r, true_positive_r)
        plt.figure(1)       
        label = '{:.1f}% neg:{} pos:{} {}'.format(roc_auc * 100, neg, pos, estimator_name)
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
'''        
def plot_roc(test_docs, test_labels, estimator, estimator_name, neg, pos,k):
        try:
            classifier_probas = estimator.decision_function(test_docs)
        except AttributeError:
            classifier_probas = estimator.predict_proba(test_docs)[:, 1]

        false_positive_r, true_positive_r, thresholds = metrics.roc_curve(test_labels, classifier_probas)
        roc_auc = metrics.auc(false_positive_r, true_positive_r)
        
        # calculate model precision-recall curve
        precision, recall, _ = precision_recall_curve(test_labels, classifier_probas)
        
        auc_score = auc(recall, precision)
        print('Auc_score %.3f   ' % auc_score, estimator_name)
        # plot the model precision-recall curve
        label = '{:.1f}%  {}'.format(auc_score * 100, estimator_name)
        #label = '   {}'.format( estimator_name)
        pyplot.figure(k)
        pyplot.plot(recall, precision, marker='.', label=label)
        # axis labels
        pyplot.title('precision_recall_curve')
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        pyplot.savefig(estimator_name+"PR.png", dpi=300, bbox_inches='tight')
        # show the plot
        #pyplot.show()

def load_vectors( binary=True):
        w2v_model = KeyedVectors.load('C:/Users/ASUS/Documents/Thesis/Code/Python/aravec-master/full_grams_sg_300_twitter/full_grams_sg_300_twitter.mdl')
        voc=w2v_model.wv.vocab
        w2v_model.init_sims(replace=True)  # to save memory
        vocab, vector_dim = w2v_model.wv.syn0.shape
        return w2v_model, vector_dim
# Main Run
        
print('loading training data')    
        
#train_data, train_labels = load_data("20ng-train-no-stop.txt")
#train_data, train_labels = load_csv_data("D:\MyProject\CleanL-HSAB-AbusHateTrain.csv")
#train_data, train_labels = load_csv_data("D:\\MyProject\\Cleantrain2300NR.csv")
print('loading testing data')
#test_data, test_labels = load_data("20ng-test-no-stop.txt")
#test_data, test_labels = load_csv_data("D:\MyProject\CleanL-HSAB-AbusHateTest.csv")
#test_data, test_labels = load_csv_data("D:\MyProject\\cleantest500-2.csv")
# train_data, train_labels = load_csv_data("D:\\MyProject\\CleanL-ardataset .csv")

test_data, test_labels = load_csv_data("C:/Users/ASUS/Documents/Thesis/Code/Python/SoureCode/CleanL-OSACT2020-sharedTask-dev.csv")
train_data, train_labels = load_csv_data("C:/Users/ASUS/Documents/Thesis/Code/Python/SoureCode/CleanL-OSACT2020-sharedTask-train.csv")

#test_data, test_labels = load_data("20ng-test-no-stop.txt")



x_train = np.asarray(train_data)
x_test = np.asarray(test_data)

#choose cross-valedition

#kf = KFold(n_splits=5)
# kf = StratifiedKFold(n_splits=5,random_state=100)
#kf=LeaveOneOut()
kf= ShuffleSplit(n_splits=5,random_state=100)

'''
print("--------tf features-----------")
for train_idx, test_idx in kf.split(x_train):
    tf_classify(x_train[train_idx].tolist(), train_labels[train_idx], test_data, test_labels)

print("--------tf_idf features-----------")
for train_idx, test_idx in kf.split(x_train):
    tf_idf_classify(x_train[train_idx].tolist(), train_labels[train_idx], x_train[test_idx].tolist(), train_labels[test_idx])
'''
print("--------word2vec features-----------")
#tunning
#word2vec_classifyGridSearchCV(x_train, train_labels, x_test, test_labels)
#word2vec_classifyGridSearchCV(train_data, train_labels, test_data, test_labels)

for train_idx, test_idx in kf.split(x_train,train_labels):    
    word2vec_classify(x_train[train_idx].tolist(), train_labels[train_idx],x_test, test_labels, x_train[test_idx].tolist(), train_labels[test_idx])

input('05')
no_topics =  [2]

dictionary, corpus = create_corpus(train_data + test_data)

#print(corpus)
#input('06corpus')
#print(dictionary)
#input('07diction')
time = np.zeros(len(no_topics))
topics_accu_KNN = np.zeros(len(no_topics))
topics_accu_SVM = np.zeros(len(no_topics))
topics_accu_RandomForest = np.zeros(len(no_topics))
words_accu_KNN = np.zeros(len(no_topics))
words_accu_SVM = np.zeros(len(no_topics))
words_accu_RandomForest = np.zeros(len(no_topics))
print(no_topics)
print("--------LDA features-----------")
for i in range(len(no_topics)):
	
	# train LDA model
	start = timeit.default_timer()
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=no_topics[i], id2word = dictionary, passes=10,random_state=100)
	end = timeit.default_timer()
	time[i] = end - start
	print(time[i])
	print("Word_LDA")
	test_docs = word_transform(test_data, ldamodel, dictionary,no_topics[i])
	train_docs = word_transform(train_data, ldamodel, dictionary,no_topics[i])
	xw=[]
	yw=[]
	xw1=[]
	yw1=[]

	for L in  train_docs:
		xw.append(L[0])
		yw.append(L[1])
	maxX=max(xw) 
	maxY=max(yw)
	
	for k in  xw:
		xw1.append(k/maxX)
	for k in  yw:
		yw1.append(k/maxY)
	#print(train_docs[1:4] )
	#words_accu_KNN[i], words_accu_SVM[i],words_accu_RandomForest[i] =  LDA_classify(train_docs, train_labels, test_docs, test_labels)
	print("topic_LDA")
	# get data represented as a distribution over the topics
	test_docs = topic_transform(test_data, ldamodel, dictionary,no_topics[i])
	train_docs = topic_transform(train_data, ldamodel, dictionary,no_topics[i])
	xt=[]
	yt=[]
	for L in  train_docs:
		xt.append(L[0])
		yt.append(L[1])
		
	#topics_accu_KNN[i], topics_accu_SVM[i],topics_accu_RandomForest[i] =  LDA_classify(train_docs, train_labels, test_docs, test_labels)

#print("words_accu_KNN",words_accu_KNN)
#print("words_accu_SVM",words_accu_SVM) 
#print("words_accu_RandomForest",words_accu_RandomForest)
#print("topics_accu_KNN",topics_accu_KNN)
#print("topics_accu_SVM",topics_accu_SVM)
#print("topics_accu_RandomForest",topics_accu_RandomForest) 
 
x = np.asarray(train_docs)
y=train_labels
    
# blank lists to store predicted values and actual values
predicted_y = []
expected_y = []
classifier = RandomForestClassifier(n_estimators=300)#GaussianNB()#SVC(kernel='sigmoid')#GaussianNB()
skf = StratifiedKFold(n_splits=5,random_state=100)
kfold= KFold(n_splits=5,random_state=100)
Shufflefold= ShuffleSplit(n_splits=5,test_size=0.2,random_state=100)
resultes=cross_val_score(classifier,train_docs,train_labels,cv=kfold)
print(resultes)
print('kfold',resultes.mean())

resultess=cross_val_score(classifier,train_docs,train_labels,cv=skf)
print(resultess)
print('StratifiedKFold',resultess.mean())
input('08start tuninig')
resultesss=cross_val_score(classifier,train_docs,train_labels,cv=Shufflefold)
'''
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestClassifier(n_estimators=100))
hyperparameters = { 'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],
                    'randomforestclassifier__max_depth': [None, 1, 3, 5]}
'''	
pipeline = make_pipeline(preprocessing.StandardScaler(),CatBoostClassifier(depth=6,border_count=32,random_state=420,loss_function='Logloss' ,task_type="GPU"))
hyperparameters = { 'catboostclassifier__iterations' : [250,100,500],'catboostclassifier__learning_rate':[0.03,0.001,0.01,0.1,0.3]}
estemater=CatBoostClassifier(depth=5,border_count=32,random_state=420,loss_function='Logloss' ,task_type="GPU")
hyperparameters = { 'iterations' : [250,100,500],'learning_rate':[0.03,0.001,0.01,0.1,0.05]}
skf= ShuffleSplit(n_splits=5,random_state=100)
clf = GridSearchCV(estimator=estemater,param_grid=hyperparameters,scoring ='roc_auc',cv = 5,refit = "roc_auc",return_train_score="True")

#clf = GridSearchCV(pipeline, hyperparameters, cv=5,scoring = 'roc_auc')
clf.fit(x, y)
print('clf',clf.best_params_)
print(resultesss)
print('ShuffleSplitFold',resultesss.mean())
pred = clf.predict(test_docs)
best_model = clf.best_estimator_
pred = best_model.predict(test_docs)
negativetest = len(best_model.predict(test_docs)[best_model.predict(test_docs) == 0])
positivetest = len(best_model.predict(test_docs)[best_model.predict(test_docs) == 1])
plot_auc(test_docs, test_labels, best_model, clf.__class__.__name__+"_topic_test", negativetest, positivetest,1)
plot_roc(test_docs, test_labels, best_model,clf.__class__.__name__+"_topic_test", negativetest, positivetest,3)

input('09end tuninig')
for train_index, test_index in Shufflefold.split(x, y):
        # specific ".loc" syntax for working with dataframes
        
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # create and fit classifier
        classifier = RandomForestClassifier(max_depth=1,n_estimators=100,max_features='auto')#SVC(kernel='sigmoid')#GaussianNB()
        classifier.fit(x_train, y_train)

        # store result from classification
        predicted_y.extend(classifier.predict(x_test))

        # store expected result for this specific fold
        expected_y.extend(y_test)
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        print("Accuracy: " + accuracy.__str__())
        predicted_=classifier.predict(test_docs)
        accuracy = metrics.accuracy_score(test_labels, predicted_)
        print("Accuracy test----------------------: " + accuracy.__str__())
        classifier_name = classifier.__class__.__name__
        print('classifier_name',classifier_name)
        
        negative = len(classifier.predict(test_docs)[classifier.predict(test_docs) == 0])
        positive = len(classifier.predict(test_docs)[classifier.predict(test_docs) == 1])

        plot_auc(test_docs,test_labels,classifier, classifier_name, negative, positive,6)
        plot_roc(test_docs,test_labels,classifier, classifier_name, negative, positive,7)
        '''
        negative = len(classifier.predict(x_test)[classifier.predict(x_test) == 0])
        positive = len(classifier.predict(x_test)[classifier.predict(x_test) == 1])

        plot_auc(x_test,y_test,classifier, classifier_name, negative, positive)
        plot_roc(x_test,y_test,classifier, classifier_name, negative, positive)
        '''

# save and print accuracy
       
#predicted_=classifier.predict(test_docs)

#accuracy = metrics.accuracy_score(test_labels, predicted_)
#print("Accuracy test: " + accuracy.__str__())
print('kfold',resultes.mean()*100)
print('StratifiedKFold',resultess.mean()*100)
print('ShuffleSplitFold',resultesss.mean()*100)

#start_time = time.time() ## pomiar czasu: start pomiaru czasu
#print(time.ctime()) 

