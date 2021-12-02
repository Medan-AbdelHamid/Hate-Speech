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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import processData
import HateSpeechDetect
from imblearn.over_sampling import(SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,KMeansSMOTE)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet

LOG_HEAD = '[%(asctime)s] %(levelname)s: %(message)s'
basicConfig(format=LOG_HEAD, level=INFO)

def label_smoothing(inputs, epsilon = 0.1):
	'''
	Implement label smoothing

	Args:
		inputs: [Tensor], A 3d tensor with shape of [N, T, V]
		epsilon: [Float], Smoothing rate

	Return:
		A tensor after smoothing
	'''

	K = 2#inputs.get_shape().as_list()[-1]
	return ((1 - epsilon) * inputs) + (epsilon / K) 

class HateSpeechDetection(object):
    """
    :HateSpeechDetection class.
    :we try to build a model through several classifiers
    :we may run soft voting or a special ensembling to enhance metrics.
    """
    def __init__(self, 
                 embeddings_file=None, 
                 datatype="B",
                 dataset_file=None, 
                 plot_roc=False, 
                 split=0.80, 
                 detailed=False, 
                 runTest=False,
                 voting_type=0):
        """
        :param embeddings_file: path to the embeddings file.
        :param dataset_file: path to a pre-labeled dataset file.
        :param plot_roc: boolean, plot ROC figure.
        :param split: float, data split fraction i.e. train | test split (default: 80% | 10%)
        :param detailed: boolean, output classifiers' parameters info i.e. name, parameters' value, .. etc.
        """

        plt.clf()
        self.datatype = datatype
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
        # vectorizer = TfidfVectorizer()
        # self.x = vectorizer.fit(train_txt)
        # print(vectorizer.get_feature_names())
        # print(self.x.shape)
        # print(vectorizer.idf_)
        # return
        train_vectors = self.average_feature_vectors(train_tokens, 'training')
        test_vectors = self.average_feature_vectors(test_tokens, 'testing')
        # vectorized features
        self.X_train = self.remove_nan(train_vectors)
        self.X_test = self.remove_nan(test_vectors)

        print('Original dataset shape %s' % Counter(self.y_train))
        if not runTest:
            sm = SMOTE(random_state=42)
            self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
            print('Resampled dataset shape %s' % Counter(self.y_train))

        self.accuracies = {}
        self.cmatrix = {}
        self.predictions = dict()
        self.predict_proba = dict()
        if voting_type == 1:
            clssf = processData.classifiers
        else:
            clssf = processData.theClassifiers

        if self.isTest:
            # -4- fit classifiers
            for c in clssf:
                self.cmatrix[c.__class__.__name__] = self.predict(c, plot_roc)
        else:
            for c in clssf:
                self.cmatrix[c.__class__.__name__] = self.classify(c, plot_roc)
                
        if voting_type == 2:
            self.l2 = np.matrix([
                self.predictions['MLPClassifier-'+self.datatype].tolist(),
                self.predictions['RandomForestClassifier-'+self.datatype].tolist(),
                self.predictions['CatBoostClassifier-'+self.datatype].tolist(),
                self.predictions['XGBClassifier-'+self.datatype].tolist(),
                self.predictions['SVC-B'].tolist(),
                                ]).T
            level2 = np.squeeze(np.asarray(self.l2))
            self.dataset_level_two = np.hstack((level2, np.array([self.y_test]).T))
            self.Level2HateSpeech = HateSpeechDetect.Level2HateSpeechDetection(dataset=self.dataset_level_two,
                                                                               split=self.split, 
                                                                               is_Test=self.isTest)

        avg_f1 = 0
        allresults = 'Results: '
        # info('results ...')
        for k, v in self.accuracies.items():
            rstring = '\tMacAvg. {:.2f}% F1. {:.2f}% P. {:.2f} R. {:.2f} Acc. {:.2f}%: {}'
            cres = rstring.format(v[0] * 100, v[1] * 100, v[2] * 100, v[3] * 100,v[4] * 100, k) 
            # print(cres)
            allresults += '\n' + cres
            avg_f1 += float(v[1])
            
        for k, v in self.cmatrix.items():
            rstring = '{}\n----------------------\n{}'
            cres = rstring.format(k, v) 
            # print(cres)
            allresults += '\n' + cres

        processData.add_test_to_database(allresults, 'classifiers_params')
        # print('OVERALL avg F1 test {:.2f}%'.format((avg_f1 / len(self.accuracies)) * 100))
        if voting_type == 2:
            y_pred = pd.Series( [1 if sum(row)>0 else 0 for row in self.l2.A])
            confusion_matrix= metrics.confusion_matrix(self.y_test, y_pred)
            print('---------------avereage confusion----------------')
            print(confusion_matrix)
        else:
            if self.isTest:
                lr_data = np.array([
                    self.predictions['RandomForestClassifier-'+self.datatype],
                    self.predictions['MLPClassifier-'+self.datatype],
                    self.predictions['XGBClassifier-'+self.datatype],
                    self.predictions['CatBoostClassifier-'+self.datatype],
                    self.predictions['SVC-'+self.datatype]
                ]).transpose()
                # lr_data = np.array([
                #     self.predict_proba['RandomForestClassifier-'+self.datatype][:,1],
                #     self.predict_proba['MLPClassifier-'+self.datatype][:,1],
                #     self.predict_proba['XGBClassifier-'+self.datatype][:,1],
                #     self.predict_proba['CatBoostClassifier-'+self.datatype][:,1],
                #     self.predict_proba['SVC-'+self.datatype][:,1]
                # ]).transpose()

                with open('Results/LogisticRegressionVoting', 'rb') as f:
                    LinearRegressionVotingCls = pickle.load(f)
                coeff = LinearRegressionVotingCls.coef_[0]
                y_pred = LinearRegressionVotingCls.predict(lr_data)
                predict_proba = LinearRegressionVotingCls.predict_proba(lr_data)
                self.predict_proba['LgR_VOTES'] = predict_proba[:,1]
                self.predictions['LgR_VOTES'] = y_pred
                confusion_matrix= metrics.confusion_matrix(self.y_test, y_pred)
                print('---------------LgR Vote confusion----------------')
                print(confusion_matrix)
                negative = len(y_pred[y_pred == 0])
                positive = len(y_pred[y_pred == 1])
                if plot_roc:
                    info('plotting roc of ... LgR VOTES')
                    self.plot_auc(None, 'LgR_VOTES', negative, positive)

                with open('Results/LinearRegressionVoting', 'rb') as f:
                    LinearRegressionVotingCls = pickle.load(f)
                coeff = LinearRegressionVotingCls.coef_
                y_pred = LinearRegressionVotingCls.predict(lr_data)

                with open('Results/ElasticNetVoting', 'rb') as f:
                    LinearRegressionVotingCls = pickle.load(f)
                coeff = LinearRegressionVotingCls.coef_
                y_pred = LinearRegressionVotingCls.predict(lr_data)

                coeff = [0.22, 0.26, 0.28, 0.27, 0.37]
                coeff = [0.31, 0.42, 0., 0., 0.27]
                predict_proba = sum([
                                        coeff[0] * self.predict_proba['RandomForestClassifier-'+self.datatype][:,1],
                                        coeff[1] * self.predict_proba['MLPClassifier-'+self.datatype][:,1],
                                        coeff[2] * self.predict_proba['XGBClassifier-'+self.datatype][:,1],
                                        coeff[3] * self.predict_proba['CatBoostClassifier-'+self.datatype][:,1],
                                        coeff[4] * self.predict_proba['SVC-'+self.datatype][:,1]])
                newy_pred = predict_proba
                predict_proba = sum([
                                        coeff[0] * self.predictions['RandomForestClassifier-'+self.datatype],
                                        coeff[1] * self.predictions['MLPClassifier-'+self.datatype],
                                        coeff[2] * self.predictions['XGBClassifier-'+self.datatype],
                                        coeff[3] * self.predictions['CatBoostClassifier-'+self.datatype],
                                        coeff[4] * self.predictions['SVC-'+self.datatype]])
                y_pred = pd.Series( [1 if row>=0.4 else 0 for row in predict_proba])
                # newy_pred = sum([
                # coeff[0] * self.predict_proba['RandomForestClassifier-'+self.datatype][:,1],
                # coeff[1] * self.predict_proba['MLPClassifier-'+self.datatype][:,1],
                # coeff[2] * self.predict_proba['XGBClassifier-'+self.datatype][:,1],
                # coeff[3] * self.predict_proba['CatBoostClassifier-'+self.datatype][:,1],
                # coeff[4] * self.predict_proba['SVC-'+self.datatype][:,1]
                # ])
                self.predict_proba['MS_VOTES'] = newy_pred
                self.predictions['MS_VOTES'] = y_pred
                confusion_matrix= metrics.confusion_matrix(self.y_test, y_pred)
                print('---------------MSoft confusion----------------')
                print(confusion_matrix)
                negative = len(y_pred[y_pred == 0])
                positive = len(y_pred[y_pred == 1])
                if plot_roc:
                    info('plotting roc of ... MSoft VOTES')
                    self.plot_auc(None, 'MS_VOTES', negative, positive)            

                newy_pred = sum([
                0.2 * self.predictions['RandomForestClassifier-'+self.datatype],
                0.2 * self.predictions['MLPClassifier-'+self.datatype],
                0.2 * self.predictions['XGBClassifier-'+self.datatype],
                0.2 * self.predictions['CatBoostClassifier-'+self.datatype],
                0.2 * self.predictions['SVC-'+self.datatype]
                ])
                y_pred = pd.Series([1 if row >= 0.4 else 0 for row in newy_pred])
                self.predict_proba['MH_VOTES'] = newy_pred
                self.predictions['MH_VOTES'] = y_pred
                confusion_matrix= metrics.confusion_matrix(self.y_test, y_pred)
                print('---------------mHard confusion----------------')
                print(confusion_matrix)
                negative = len(y_pred[y_pred == 0])
                positive = len(y_pred[y_pred == 1])
                if plot_roc:
                    info('plotting roc of ... MH_VOTES VOTES')
                    self.plot_auc(None, 'MH_VOTES', negative, positive)            
            else:
                # lr_data = np.array([
                #     self.predict_proba['RandomForestClassifier-'+self.datatype][:,1],
                #     self.predict_proba['MLPClassifier-'+self.datatype][:,1],
                #     self.predict_proba['XGBClassifier-'+self.datatype][:,1],
                #     self.predict_proba['CatBoostClassifier-'+self.datatype][:,1],
                #     self.predict_proba['SVC-'+self.datatype][:,1]
                # ]).transpose()
                lr_data = np.array([
                    self.predictions['RandomForestClassifier-'+self.datatype],
                    self.predictions['MLPClassifier-'+self.datatype],
                    self.predictions['XGBClassifier-'+self.datatype],
                    self.predictions['CatBoostClassifier-'+self.datatype],
                    self.predictions['SVC-'+self.datatype]
                ]).transpose()
                best_fit(lr_data, self.y_test)

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
        train_df, test_df = train_test_split(dataset, train_size=self.split,random_state=20021204)
        # t = pd.DataFrame(train_df, columns=['NB', 'ANOMALY', 'FULLTEXT'])
        # t.to_csv('thesis-trainset.csv')
        # s = pd.DataFrame(test_df, columns=['NB', 'ANOMALY', 'FULLTEXT'])
        # s.to_csv('thesis-testset.csv')
        if self.showInfo:
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
        classifier_name = classifier.__class__.__name__ + "-" + self.datatype
        try:
            classifier_name += '-' + classifier.get_params()['voting']
        except Exception:
            pass
        info('classifier name... {}'.format(classifier_name))
        print('classifier_name',classifier_name)
        if self.showInfo:
            info('fitting data ...')
            info('\n\ncreated \n\n{}'.format(classifier))

        # classifier.fit(self.X_train, self.y_train)
        # # now you can save it to a file
        # with open('Results/'+classifier_name, 'wb') as f:
        #     pickle.dump(classifier, f)
        with open('Results/'+classifier_name, 'rb') as f:
            classifier = pickle.load(f)

        pscore = classifier.score(self.X_test, self.y_test)
        if self.showInfo:
            info('\n\t{}() ACCURACY: {}\n'.format(classifier_name, pscore))

        # F1 score
        # y_pred = classifier.predict(self.X_test)
        predict_proba    = classifier.predict_proba(self.X_test)
        y_pred = (predict_proba[:,1] >= 0.40).astype(bool) # set threshold as 0.4
        self.predictions[classifier_name] = y_pred
        self.predict_proba[classifier_name] = predict_proba
        # self.predictions[classifier_name] = y_pred
        f1_score = metrics.f1_score(self.y_test, y_pred, zero_division=0)

        # macro accuracy (macro average)
        macc = metrics.f1_score(self.y_test, y_pred, pos_label=None, average='macro', zero_division=0)

        # precision and recall
        print(metrics.classification_report(self.y_test,  y_pred))
        confusion_matrix= metrics.confusion_matrix(self.y_test,  y_pred)
        print('---------------confusion----------------')
        print(confusion_matrix)
        recall = metrics.recall_score(self.y_test, y_pred)
        precision = metrics.precision_score(self.y_test, y_pred)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        results = [macc, f1_score, precision, recall, accuracy]
        self.accuracies[classifier_name] = results

        # prediction
        negative = len(classifier.predict(self.X_test)[classifier.predict(self.X_test) == 0])
        positive = len(classifier.predict(self.X_test)[classifier.predict(self.X_test) == 1])

        if plot_roc:
            # info('plotting roc of ... {}'.format(classifier_name))
            self.plot_auc(classifier, classifier_name, negative, positive)
            # self.plot_roc(classifier, classifier_name, negative, positive)
        return confusion_matrix

    def predict(self, classifier=None, plot_roc=True):
        classifier_name = classifier.__class__.__name__ + "-" + self.datatype
        try:
            classifier_name += '-' + classifier.get_params()['voting']
        except Exception:
            pass
        print('classifier_name',classifier_name)
        # load the classifier
        with open('Results/'+classifier_name, 'rb') as f:
            classifier = pickle.load(f)
         
        # print(len(self.X_test),len(self.y_test))
        pscore = classifier.score(self.X_test, self.y_test)
        # pscore = classifier.score(self.X_train, self.y_train)
        if self.showInfo:
            info('\n\n\t{}() ACCURACY: {}\n'.format(classifier_name, pscore))

        # F1 score
        y_pred = classifier.predict(self.X_test)
        predict_proba = classifier.predict_proba(self.X_test)
        # y_pred = (predict_proba[:,1] >= 0.4).astype(bool) # set threshold as 0.35
        self.predictions[classifier_name] = y_pred
        self.predict_proba[classifier_name] = predict_proba
        f1_score = metrics.f1_score(self.y_test, y_pred, zero_division=0)

        # macro accuracy (macro average)
        macc = metrics.f1_score(self.y_test, y_pred, pos_label=None, average='macro', zero_division=0)
        confusion_matrix = metrics.confusion_matrix(self.y_test,  y_pred)
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
            info('plotting roc of ... {}'.format(classifier_name))
            self.plot_auc(classifier, classifier_name, negative, positive)            
            
        return confusion_matrix

    def plot_auc(self, estimator, estimator_name, neg, pos):
        try:
            if estimator == None:
                classifier_probas = self.predict_proba[estimator_name]
            else:
                try:
                    classifier_probas = estimator.decision_function(self.X_test)
                except AttributeError:
                    classifier_probas = estimator.predict_proba(self.X_test)[:, 1]
        
            false_positive_r, true_positive_r, thresholds = metrics.roc_curve(self.y_test, classifier_probas)
            roc_auc = metrics.auc(false_positive_r, true_positive_r)
        
            label = '{:.1f}% neg:{} pos:{} {}'.format(roc_auc * 100, neg, pos, estimator_name)
            roc_path = 'Results/ROC-{}-{}-{}.png'.format(self.test_indx, self.dataset_file, self.datatype)
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
        except AttributeError:
            print('estimator_name: error print')

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
            precision, recall, _ = precision_recall_curve(self.y_train, classifier_probas)
        
        roc_path = 'Results/ROC-{}-{}-{}.png'.format(self.test_indx, self.dataset_file, self.datatype)
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
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        # show the plot
        #pyplot.show()

    @staticmethod
    def remove_nan(x):
        """remove NaN values from data vectors"""
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        x_clean = imp.fit_transform(x)
        return x_clean

def best_fit(X, Y):
    for model in [LogisticRegression(fit_intercept=True, l1_ratio = 0.1, penalty='elasticnet', solver='saga'), 
                  ElasticNet(fit_intercept=True, l1_ratio = 0, max_iter=1000),
                  LinearRegression(fit_intercept=True)
                  ]:
        model.fit(X, Y)
        modelName = model.__class__.__name__ + 'Voting'
        print('Model Coeff {} = {} ---{}'.format(modelName, model.coef_, model.intercept_))
        savepath = 'Results/' + modelName
        with open(savepath, 'wb') as f:
            pickle.dump(model, f)

