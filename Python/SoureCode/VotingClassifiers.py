# coding: utf-8
    
"""
VotingClassifiers.py is a file to deploy Hate Speech on Arabic 
    (Levantine dataset from twitter) using Word Embeddings.
    We use voting on several classiefiers.
        gard voting.
        soft voting.
        
@author: Medyan
Date: Dec. 2020
"""
import argparse
from gensim.models import KeyedVectors
import processData
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib.pyplot import figure
from matplotlib import pyplot
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize
from sklearn.impute import SimpleImputer

figure(num=2, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
 
class StackingHSDetection(object):
    """
    :HateSpeechDetection class.
    :we try to build a model through several classifiers
    :we run voting to enhance metrics.
    """
    def __init__(self, 
                 embeddings_file=None, 
                 dataset_file=None, 
                 plot_roc=False, 
                 split=0.80, 
                 detailed=False, 
                 runTest=False,
                 voting=False):
        """
        :param embeddings_file: path to the embeddings file.
        :param dataset_file: path to a pre-labeled dataset file.
        :param plot_roc: boolean, plot ROC figure.
        :param split: float, data split fraction i.e. train | test split (default: 80% | 10%)
        :param detailed: boolean, output classifiers' parameters info i.e. name, parameters' value, .. etc.
        """
        self.test_indx = processData.get_test_seq()
        self.dataset_file = dataset_file
        self.split = split
        self.showInfo = detailed
        self.isTest = runTest

        self.embeddings, self.dimension = self.load_vectors(embeddings_file)
        # -1- read dataset
        train, test = self.read_data(self.dataset_file)
        train_txt, test_txt = train['FULLTEXT'], test['FULLTEXT']
       
        self.y_train = train['ANOMALY']
        self.y_test = test['ANOMALY']
        # -2- tokeenize data-- #
        train_tokens = self.tokenize_data(train_txt, 'training')
        test_tokens = self.tokenize_data(test_txt, 'testing')
        # -3- data representation-- #
        train_vectors = self.average_feature_vectors(train_tokens, 'training')
        test_vectors = self.average_feature_vectors(test_tokens, 'testing')
        # vectorized features
        self.X_train = self.remove_nan(train_vectors)
        self.X_test = self.remove_nan(test_vectors)

        self.accuracies = {}
        self.cmatrix = {}
        
        # get the models to evaluate
        models = self.get_models()
        # evaluate the models and store results
        results, names = list(), list()
        for name, model in models.items():
            scores = self.evaluate_model(model)
            results.append(scores)
            names.append(name)
            print('>%s -> %.3f (%.3f)---Wine dataset' % (name, mean(scores), std(scores)))
        # plot model performance for comparison
        pyplot.rcParams["figure.figsize"] = (15,6)
        pyplot.boxplot(results, labels=[s+"-wine" for s in names], showmeans=True)
        pyplot.show()

    # get a stacking ensemble of models
    def get_stacking(self):
        # define the base models
        level0 = list()
        level0.append(('rf', processData.theClassifiers[0]))
        level0.append(('sgd', processData.theClassifiers[1]))
        level0.append(('lsvc', processData.theClassifiers[2]))
        level0.append(('svc', processData.theClassifiers[3]))
        level0.append(('xgb', processData.theClassifiers[4]))
        # level0.append(('cat', processData.theClassifiers[5]))
        level0.append(('lrcv', processData.theClassifiers[6]))
        level0.append(('lr', processData.theClassifiers[7]))
    
        # define meta learner model
        level1 = processData.theClassifiers[5]
        # define the stacking ensemble
        model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
        return model
    
    # get a list of models to evaluate
    def get_models(self):
        models = dict()
        models['rf'] = processData.theClassifiers[0]
        models['sgd'] = processData.theClassifiers[1]
        models['lsvc'] = processData.theClassifiers[2]
        models['svc'] = processData.theClassifiers[3]
        models['xgb'] = processData.theClassifiers[4]
        models['cat'] = processData.theClassifiers[5]
        models['lrcv'] = processData.theClassifiers[6]
        models['lr'] = processData.theClassifiers[7]
        models['stacking'] = self.get_stacking()
        return models
    
    # evaluate a give model using cross-validation
    def evaluate_model(self, model):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, 
                                 self.X_train, 
                                 self.y_train, 
                                 scoring='accuracy', 
                                 cv=cv, 
                                 n_jobs=-1, 
                                 error_score='raise')
        return scores
 
    @staticmethod
    def load_vectors(model_name, binary=True):
        """load the pre-trained embedding model"""
        '''
        if binary:
            w2v_model = KeyedVectors.load_word2vec_format(model_name, binary=True)
        else:
             
        '''
        w2v_model = KeyedVectors.load('C:/Users/ASUS/Documents/Thesis/Code/Python/aravec-master/full_grams_sg_300_twitter/full_grams_sg_300_twitter.mdl')
        w2v_model.init_sims(replace=True)  # to save memory
        vocab, vector_dim = w2v_model.wv.syn0.shape
        #print('model.corpus_count',w2v_model.corpus_count)
        #print('model.corpus_total_words',w2v_model.corpus_total_words)
        #print('len(model.wv.vocab)',len(w2v_model.wv.vocab))
        return w2v_model, vector_dim

    def read_data(self, dataset_in):
        dataset = pd.read_csv(dataset_in)
        # split train/test
        if self.isTest:
            self.split = 0.01
        train_df, test_df = train_test_split(dataset, train_size=self.split,random_state=412)
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
        total_tokens = []
        for txt in examples_txt:
            words, num = self.tokenize(txt)
            tokens.append(words)
            total_tokens.append(num)
        
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
                # print(token)    
        np.seterr(divide='ignore', invalid='ignore')
        feature_vec = np.divide(feature_vec, retrieved_words)
        # print(retrieved_words,len(words))
        return feature_vec

    def average_feature_vectors(self, examples, type_='NaN'):
        """
        :param examples: a list of lists (each list contains words) e.g. [['hi','do'], ['you','see'], ... ]
        :param type_: (optional) type of examples text e.g. train / test
        :return: the average word vector of each list
        """

        feature_vectors = np.zeros((len(examples), self.dimension), dtype="float32")
        for i, example in enumerate(examples):
            feature_vectors[i] = self.feature(example)
        
        return feature_vectors

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
    parser.add_argument("--isVoting", help="use voting model or conventional models.")
    parser.add_argument("--isTest", help="train the model or apply the model on test dataset.")
    parser.add_argument("--plot", help="print ROC curve or not")
    
    args = parser.parse_args()
    vec = args.vectors

    # vectors file and dataset pathes
    embeddings_path = args.vectors if args.vectors else "arabic.bin"
    # dataset_path = args.dataset if args.dataset else "thesis-train.csv"
    dataset_path = args.dataset if args.dataset else "CleanL-HSAB-AbusHateTrain.csv"
    isVoting = args.isVoting if args.isVoting else False
    isTest = args.isTest if args.isTest else False
    plot = args.plot if args.plot else True
    # run
    medyan = StackingHSDetection(embeddings_path, 
                                 dataset_path, 
                                 voting=isVoting,
                                 runTest=isTest,
                                 plot_roc=plot)
