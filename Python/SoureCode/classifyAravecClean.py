# coding: utf-8

"""
asa.py is a simple (Arabic) Sentiment Analysis using Word Embeddings.
Author: Aziz Alto
Date: Aug. 2016
"""
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt    
from imblearn.over_sampling import(SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,KMeansSMOTE)
from collections import Counter
import gensim
import pickle
import re
import string
import argparse
from logging import info, basicConfig, INFO
# -- 3rd party -- 
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize
# -- classifiers -- #
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
from sklearn.svm import LinearSVC, NuSVC,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import FastText
LOG_HEAD = '[%(asctime)s] %(levelname)s: %(message)s'
basicConfig(format=LOG_HEAD, level=INFO)

def get_vec(n_model,dim, token):
    vec = np.zeros(dim)
    is_vec = False
    if token not in n_model.wv:
        _count = 0
        is_vec = True
        for w in token.split("_"):
            if w in n_model.wv:
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec = vec / _count
    else:
        vec = n_model.wv[token]
    return vec

class ArSentiment(object):
    def __init__(self, embeddings_file=None, dataset_file=None, plot_roc=False, split=0.90 , detailed=False):
        """
        :param embeddings_file: path to the embeddings file.
        :param dataset_file: path to a labeled dataset file.
        :param plot_roc: boolean, plot ROC figure.
        :param split: float, data split fraction i.e. train | test split (default: 90% | 10%)
        :param detailed: boolean, output classifiers' parameters info i.e. name, parameters' value, .. etc.
        """

        self.dataset_file = dataset_file
        self.split = split

        self.embeddings, self.dimension = self.load_vectors(embeddings_file)
        print(' self.embeddings',self.embeddings)
        #input('embedding')
        #read dataset
        train, test = self.read_data(self.dataset_file)
        print(train)     
        train_txt, test_txt = train['FULLTEXT'], test['FULLTEXT']
       
        self.y_train = train['ANOMALY']
        self.y_test = test['ANOMALY']
        # -- dataset preprocessing -- #
        train_tokens = self.tokenize_data(train_txt, 'training')
        test_tokens = self.tokenize_data(test_txt, 'testing')
        # -- vectorize training/testing data -- #
        train_vectors = self.average_feature_vectors(train_tokens, 'training')
        test_vectors = self.average_feature_vectors(test_tokens, 'testing')
        print(train_vectors)
        # vectorized features
        self.X_train = self.remove_nan(train_vectors)
        self.X_test = self.remove_nan(test_vectors)
        '''
        tweets = pd.read_csv("CleanL-HSAB-AbusHateTrain.csv")
        X = tweets['FULLTEXT']  
        y = tweets["ANOMALY"]
        processed_tweets = []
 
        for tweet in range(0, len(X)):  
        # Remove all the special characters
            processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))
            processed_tweets.append(processed_tweet)
        from sklearn.feature_extraction.text import TfidfVectorizer  
        tfidfconverter = TfidfVectorizer(ngram_range=(1,3),max_features=250, min_df=5, max_df=0.7)  
        X = tfidfconverter.fit_transform(processed_tweets).toarray()

        from sklearn.model_selection import train_test_split  
        self.X_train,self.X_test,self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=101)
        '''
        print('Original dataset shape %s' % Counter(self.y_train))
        '''
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
        print('Resampled dataset shape %s' % Counter(self.y_train))
        ''' 
        #input('ok')
        info('Done loading and vectorizing data.')
        info("--- Sentiment CLASSIFIERS ---")
        info("fitting ... ")

        # classifiers to use
        classifiers = [
            RandomForestClassifier(n_estimators=300),
            SGDClassifier(random_state=42),#(loss='log', penalty='l1'),
            LinearSVC(C=1e1),
            SVC(kernel='linear', class_weight={1: 1000}),
            #XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5,base_score=0.7, booster='gbtree',colsample_bylevel=1, gamma=0,subsample=0.5,tree_method='gpu_hist', predictor='gpu_predictor'),
            XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5),
            #XGBClassifier(base_score=0.7, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.01,max_delta_step=0, max_depth=6, min_child_weight=1, missing=None,n_estimators=100, n_jobs=1, nthread=2, random_state=0, reg_alpha=0,reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,subsample=1, verbosity=1), 
            CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,loss_function='Logloss',border_count=32 ),
            # NuSVC(),
            LogisticRegressionCV(solver='liblinear'),
            #GaussianNB(),
        ]

        self.accuracies = {}

        # RUN classifiers
        for c in classifiers:
            self.classify(c, detailed, plot_roc)
        #instantiate model and train

        avg_f1 = 0
        info('results ...')
        for k, v in self.accuracies.items():
            string = '\tMacAvg. {:.2f}% F1. {:.2f}% P. {:.2f} R. {:.2f} Acc. {:.2f}%: {}'
            print(string.format(v[0] * 100, v[1] * 100, v[2] * 100, v[3] * 100,v[4] * 100, k))
            avg_f1 += float(v[1])

        print('OVERALL avg F1 test {:.2f}%'.format((avg_f1 / len(self.accuracies)) * 100))
        info("DONE!")
        
       
    @staticmethod
    def load_vectors(model_name, binary=True):
        """load the pre-trained embedding model"""
        '''
        if binary:
            w2v_model = KeyedVectors.load_word2vec_format(model_name, binary=True)
        else:
             
        '''
        w2v_model = KeyedVectors.load('C:/Users/ASUS/Documents/Thesis/Code/Python/aravec-master/full_grams_sg_300_twitter/full_grams_sg_300_twitter.mdl')
        #w2v_model = KeyedVectors.load('D:/DoIt/aravec-master/full_grams_sg_100_twitter/full_grams_sg_100_twitter.mdl')
        #w2v_model = KeyedVectors.load('D:/DoIt/aravec-master/tweet_cbow_300/tweets_cbow_300')
        #w2v_model = KeyedVectors.load('D:/DoIt/aravec-master/full_uni_cbow_300_twitter/full_uni_cbow_300_twitter.mdl')
        #w2v_model = gensim.models.Word2Vec.load('D:/DoIt/aravec-master/full_uni_cbow_300_twitter/full_uni_cbow_300_twitter.mdl')
        #w2v_model = gensim.models.Word2Vec.load('D:/DoIt/aravec-master/AOC_Skipgram/AOC_Skipgram.mdl')
        #w2v_model = gensim.models.Word2Vec.load('arabic.bin')
        #w2v_model = gensim.models.FastText.load('FastText')
        voc=w2v_model.wv.vocab
        w2v_model.init_sims(replace=True)  # to save memory
        vocab, vector_dim = w2v_model.wv.syn0.shape
        #print('model.corpus_count',w2v_model.corpus_count)
        #print('model.corpus_total_words',w2v_model.corpus_total_words)
        #print('len(model.wv.vocab)',len(w2v_model.wv.vocab))
            # similarity between two different words
        #print("Similarity between '??????????' and '??????????'",w2v_model.wv.similarity(w1="??????????", w2="??????????"))
        #input('load')
        return w2v_model, vector_dim
    def read_data(self, dataset_in):

        dataset = pd.read_csv(dataset_in)
        
        # split train/test
        #train_df, test_df = train_test_split(dataset, train_size=self.split,random_state=101)
        train_df, test_df = train_test_split(dataset, train_size=self.split,random_state=101)
        string_ = 'dataset {} {}. Split: {} training and {} testing.'
        info(string_.format(dataset_in, dataset.shape, len(train_df), len(test_df)))

        return train_df, test_df

    @staticmethod
    def tokenize(text):
        """
        :param text: a paragraph string
        :return: a list of words
        """

        try:
            try:
                txt = unicode(text, 'utf-8')  # py2
            except NameError:
                txt = text  # py3
            words = wordpunct_tokenize(txt)
            length = len(words)
        except TypeError:
            words, length = ['NA'], 0

        return words, length

    def tokenize_data(self, examples_txt, type_='NaN'):
        tokens = []
        info('Tokenizing the {} dataset ..'.format(type_))
        total_tokens = []
        for txt in examples_txt:
            words, num = self.tokenize(txt)
            tokens.append(words)
            total_tokens.append(num)
        info(' ... total {} {} tokens.'.format(sum(total_tokens), type_))
        return tokens

    def feature(self, words):
        """average words' vectors"""

        feature_vec = np.zeros((self.dimension,), dtype="float32")
        retrieved_words = 0
        for token in words:
            try:
                feature_vec = np.add(feature_vec, self.embeddings[token])
                retrieved_words += 1
            except KeyError:
                pass  # if a word is not in the embeddings' vocabulary discard it
                print(token)    
        np.seterr(divide='ignore', invalid='ignore')
        feature_vec = np.divide(feature_vec, retrieved_words)
        print(retrieved_words,len(words))
        print('+++++++++++++++++++++++++++++++++++++++')
        return feature_vec

    def average_feature_vectors(self, examples, type_='NaN'):
        """
        :param examples: a list of lists (each list contains words) e.g. [['hi','do'], ['you','see'], ... ]
        :param type_: (optional) type of examples text e.g. train / test
        :return: the average word vector of each list
        """

        feature_vectors = np.zeros((len(examples), self.dimension), dtype="float32")
        info("Vectorizing {} tokens ..".format(type_))
        for i, example in enumerate(examples):
            feature_vectors[i] = self.feature(example)
        
        return feature_vectors

    def classify(self, classifier=None, info_=False, plot_roc=False):

        classifier_name = classifier.__class__.__name__
        print('classifier_name',classifier_name)
        if info_:
            info('fitting data ...')
            info('\n\ncreated \n\n{}'.format(classifier))

        classifier.fit(self.X_train, self.y_train)
        # now you can save it to a file
        with open(classifier_name, 'wb') as f:
            pickle.dump(classifier, f)

        # and later you can load it
        '''load 
        with open('RandomForestClassifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        ''' 
        #pscore = clf.score(self.X_test, self.y_test) 
        #print('pscore',pscore)
        pscore = classifier.score(self.X_test, self.y_test)
        
        if info_:
            info('\n\t{}() ACCURACY: {}\n'.format(classifier_name, pscore))

        # F1 score
        y_pred = classifier.predict(self.X_test)
        f1_score = metrics.f1_score(self.y_test, y_pred)

        # macro accuracy (macro average)
        macc = metrics.f1_score(self.y_test, y_pred, pos_label=None, average='macro')

        # precision and recall
        print(metrics.classification_report(self.y_test,  y_pred))
        confusion_matrix= metrics.confusion_matrix(self.y_test,  y_pred)
        print(confusion_matrix)
        #input('---------------confusion----------------')
        recall = metrics.recall_score(self.y_test, y_pred)
        precision = metrics.precision_score(self.y_test, y_pred)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        results = [macc, f1_score, precision, recall,accuracy]
        self.accuracies[classifier_name] = results

        # prediction
        negative = len(classifier.predict(self.X_test)[classifier.predict(self.X_test) == 0])
        positive = len(classifier.predict(self.X_test)[classifier.predict(self.X_test) == 1])

        if plot_roc:
            info('plotting roc of ... {}'.format(classifier_name))
            self.plot_auc(classifier, classifier_name, negative, positive)

    def plot_auc(self, estimator, estimator_name, neg, pos):
        try:
            classifier_probas = estimator.decision_function(self.X_test)
        except AttributeError:
            classifier_probas = estimator.predict_proba(self.X_test)[:, 1]

        false_positive_r, true_positive_r, thresholds = metrics.roc_curve(self.y_test, classifier_probas)
        roc_auc = metrics.auc(false_positive_r, true_positive_r)
        from sklearn.metrics import confusion_matrix


        label = '{:.1f}% neg:{} pos:{} {}'.format(roc_auc * 100, neg, pos, estimator_name)
        plt.plot(false_positive_r, true_positive_r, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('ROC score(s)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right', prop={'size': 10})
        plt.savefig("ROC000.png", dpi=300, bbox_inches='tight')
        plt.grid()
        
    @staticmethod
    def remove_nan(x):
        """remove NaN values from data vectors"""
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        x_clean = imp.fit_transform(x)
        return x_clean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", help="path a pre-trained vectors model.")
    parser.add_argument("--dataset", help="path a labeled (0/1) sentiment dataset.")

    args = parser.parse_args()
    vec = args.vectors

    # vectors file
    #embeddings_path = args.vectors if args.vectors else "embeddings/defbin.bin"
    embeddings_path = args.vectors if args.vectors else "arabic.bin"
    # dataset file
    #dataset_path = args.dataset if args.dataset else "datasets/TrainDatacleanedF1.csv"
    #dataset_path = args.dataset if args.dataset else "D:\\MyProject\\Cleantrain2300NR.csv"
    #dataset_path = args.dataset if args.dataset else "D:\MyProject\CleanL-ardataset.csv"
    dataset_path = args.dataset if args.dataset else "C:/Users/ASUS/Documents/Thesis/Code/Python/SoureCode/Cleantrain2300NRU.csv"
    # run
    medyan = ArSentiment(embeddings_path, dataset_path, plot_roc=True)
