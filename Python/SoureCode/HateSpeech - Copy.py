# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 19:08:56 2020

@author: Medyan
"""
from collections import Counter
from logging import info, basicConfig, INFO
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize
import pickle
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import processData
import HateSpeechDetect

LOG_HEAD = '[%(asctime)s] %(levelname)s: %(message)s'
basicConfig(format=LOG_HEAD, level=INFO)

class HateSpeechDetection(object):
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

        plt.clf()
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

        print('Original dataset shape %s' % Counter(self.y_train))

        self.accuracies = {}
        self.cmatrix = {}
        self.predictions = dict()

        if self.isTest:
            # -4- fit classifiers
            if voting:
                for c in processData.classifiers:
                    self.cmatrix[c.__class__.__name__] = self.predict(c, plot_roc)
            else:
                for c in processData.theClassifiers:
                    self.cmatrix[c.__class__.__name__] = self.predict(c, plot_roc)
        else:
            if voting:
                for c in processData.classifiers:
                    self.cmatrix[c.__class__.__name__] = self.classify(c, plot_roc)
            else:
                for c in processData.theClassifiers:
                    self.cmatrix[c.__class__.__name__] = self.classify(c, plot_roc)
                
                l2 = np.matrix([self.predictions['RandomForestClassifier'].tolist(),
                                self.predictions['SGDClassifier'].tolist(),
                                self.predictions['LinearSVC'].tolist(),
                                self.predictions['SVC'].tolist(),
                                self.predictions['XGBClassifier'].tolist(),
                                self.predictions['CatBoostClassifier'].tolist(),
                                self.predictions['LogisticRegressionCV'].tolist(),
                                self.predictions['LogisticRegression'].tolist(),
                                self.predictions['GaussianNB'].tolist()]).T
                level2 = np.squeeze(np.asarray(l2))
                self.dataset_level_two = np.hstack((level2, np.array([self.y_test]).T))
                self.Level2HateSpeech = HateSpeechDetect.Level2HateSpeechDetection(dataset=self.dataset_level_two,split=0.5)

        avg_f1 = 0
        allresults = 'Results: '
        # info('results ...')
        for k, v in self.accuracies.items():
            rstring = '\tMacAvg. {:.2f}% F1. {:.2f}% P. {:.2f} R. {:.2f} Acc. {:.2f}%: {}'
            cres = rstring.format(v[0] * 100, v[1] * 100, v[2] * 100, v[3] * 100,v[4] * 100, k) 
            print(cres)
            allresults += '\n' + cres
            avg_f1 += float(v[1])
            
        for k, v in self.cmatrix.items():
            rstring = '{}\n----------------------\n{}'
            cres = rstring.format(k, v) 
            allresults += '\n' + cres

        processData.add_test_to_database(allresults, 'classifiers_params')
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
        w2v_model.init_sims(replace=True)  # to save memory
        vocab, vector_dim = w2v_model.wv.syn0.shape
        #print('model.corpus_count',w2v_model.corpus_count)
        #print('model.corpus_total_words',w2v_model.corpus_total_words)
        #print('len(model.wv.vocab)',len(w2v_model.wv.vocab))
        return w2v_model, vector_dim

    def read_data(self, dataset_in):
        dataset = pd.read_csv(dataset_in)
        if self.isTest:
            # it test. no need for split
            # run prediction on the whole test dataset
            return dataset, dataset

        # split train/test
        train_df, test_df = train_test_split(dataset, train_size=self.split,random_state=412)
        string_ = 'dataset {} {}. Split: {} training and {} testing.'
        if self.showInfo:
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
        info("Vectorizing {} tokens ..".format(type_))
        for i, example in enumerate(examples):
            feature_vectors[i] = self.feature(example)
        
        return feature_vectors

    def classify(self, classifier=None, plot_roc=False):
        classifier_name = classifier.__class__.__name__
        print('classifier_name',classifier_name)
        if self.showInfo:
            info('fitting data ...')
            info('\n\ncreated \n\n{}'.format(classifier))

        classifier.fit(self.X_train, self.y_train)
        # now you can save it to a file
        with open('Results/'+classifier_name, 'wb') as f:
            pickle.dump(classifier, f)

        pscore = classifier.score(self.X_test, self.y_test)
        
        if self.showInfo:
            info('\n\t{}() ACCURACY: {}\n'.format(classifier_name, pscore))

        # F1 score
        y_pred = classifier.predict(self.X_test)
        self.predictions[classifier_name] = y_pred
        # print(y_pred.shape)
        f1_score = metrics.f1_score(self.y_test, y_pred, zero_division=0)

        # macro accuracy (macro average)
        macc = metrics.f1_score(self.y_test, y_pred, pos_label=None, average='macro', zero_division=0)

        # precision and recall
        # print(metrics.classification_report(self.y_test,  y_pred))
        confusion_matrix= metrics.confusion_matrix(self.y_test,  y_pred)
        print('---------------confusion----------------')
        print(confusion_matrix)
        recall = metrics.recall_score(self.y_test, y_pred)
        precision = metrics.precision_score(self.y_test, y_pred)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        results = [macc, f1_score, precision, recall,accuracy]
        self.accuracies[classifier_name] = results

        # prediction
        negative = len(classifier.predict(self.X_test)[classifier.predict(self.X_test) == 0])
        positive = len(classifier.predict(self.X_test)[classifier.predict(self.X_test) == 1])

        if plot_roc:
            # info('plotting roc of ... {}'.format(classifier_name))
            self.plot_auc(classifier, classifier_name, negative, positive)
        return confusion_matrix

    def predict(self, classifier=None, plot_roc=True):
        classifier_name = classifier.__class__.__name__
        print('classifier_name',classifier_name)
        # load the classifier
        with open('Results/'+classifier_name, 'rb') as f:
            classifier = pickle.load(f)
         
        print(len(self.X_test),len(self.y_test))
        pscore = classifier.score(self.X_test, self.y_test)
        # pscore = classifier.score(self.X_train, self.y_train)
        if self.showInfo:
            info('\n\n\t{}() ACCURACY: {}\n'.format(classifier_name, pscore))

        # F1 score
        y_pred = classifier.predict(self.X_test)
        print(len(y_pred))
        
        output = pd.DataFrame(data = { "Psentiment":y_pred, "Tsentiment": self.y_test})
        fileName = classifier_name+"pred"  + ".csv"
        output.to_csv(fileName, index = False, quoting = 3 )

        f1_score = metrics.f1_score(self.y_test, y_pred, zero_division=0)
        # macro accuracy (macro average)
        macc = metrics.f1_score(self.y_test, y_pred, pos_label=None, average='macro', zero_division=0)
        # precision and recall
        confusion_matrix= metrics.confusion_matrix(self.y_test,  y_pred)
        #print(self.y_test,  y_pred)
        print(confusion_matrix)
        
        recall = metrics.recall_score(self.y_test, y_pred)
        precision = metrics.precision_score(self.y_test, y_pred)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        results = [macc, f1_score, precision, recall,accuracy]
        self.accuracies[classifier_name] = results
        average_precision = average_precision_score(self.y_test,  y_pred)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision* 100))

        # prediction
        negative = len(classifier.predict(self.X_test)[classifier.predict(self.X_test) == 0])
        positive = len(classifier.predict(self.X_test)[classifier.predict(self.X_test) == 1])

        if plot_roc:
            info('plotting roc of ... {}'.format(classifier_name))
            self.plot_auc(classifier, classifier_name, negative, positive)
            # self.plot_roc(classifier, classifier_name, negative, positive)
        return confusion_matrix

    def plot_auc(self, estimator, estimator_name, neg, pos):
        try:
            classifier_probas = estimator.decision_function(self.X_test)
        except AttributeError:
            classifier_probas = estimator.predict_proba(self.X_test)[:, 1]

        false_positive_r, true_positive_r, thresholds = metrics.roc_curve(self.y_test, classifier_probas)
        roc_auc = metrics.auc(false_positive_r, true_positive_r)

        label = '{:.1f}% neg:{} pos:{} {}'.format(roc_auc * 100, neg, pos, estimator_name)
        roc_path = 'Results/ROC{}.png'.format(self.test_indx)
        plt.plot(false_positive_r, true_positive_r, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('ROC score(s)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right', prop={'size': 10})
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.grid()
        
    def plot_roc(self, estimator, estimator_name, neg, pos):
        try:
            classifier_probas = estimator.decision_function(self.X_train)
        except AttributeError:
            classifier_probas = estimator.predict_proba(self.X_train)[:, 1]

        false_positive_r, true_positive_r, thresholds = metrics.roc_curve(self.y_train, classifier_probas)
        # roc_auc = metrics.auc(false_positive_r, true_positive_r)
        
        # calculate model precision-recall curve
        if self.isTest:
            precision, recall, _ = precision_recall_curve(self.y_train, classifier_probas)
        else:
            precision, recall, _ = precision_recall_curve(self.y_test, classifier_probas)
        
        auc_score = auc(recall, precision)
        print('Auc_score %.3f   ' % auc_score, estimator_name)
        # plot the model precision-recall curve
        label = '{:.1f}%  {}'.format(auc_score * 100, estimator_name)
        #label = '   {}'.format( estimator_name)
        plt.figure(2)
        plt.plot(recall, precision, marker='.', label=label)
        # axis labels
        plt.title('precision_recall_curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        plt.savefig("lAUC.png", dpi=300, bbox_inches='tight')
        # show the plot
        #pyplot.show()

    @staticmethod
    def remove_nan(x):
        """remove NaN values from data vectors"""
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        x_clean = imp.fit_transform(x)
        return x_clean

