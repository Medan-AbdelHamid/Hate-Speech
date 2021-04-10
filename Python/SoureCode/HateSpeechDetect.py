# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 19:08:56 2020

@author: Medyan
"""
import processData
from logging import info
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics

class Level2HateSpeechDetection(object):
    """
    :param embeddings_file: path to the embeddings file.
    :param dataset_file: path to a pre-labeled dataset file.
    :param plot_roc: boolean, plot ROC figure.
    :param split: float, data split fraction i.e. train | test split (default: 80% | 10%)
    :param detailed: boolean, output classifiers' parameters info i.e. name, parameters' value, .. etc.
    """
    def __init__(self, 
                 dataset=None, 
                 plot_roc=True, 
                 split=0.80, 
                 is_Test=False,
                 detailed=False):
        """
        :param embeddings_file: path to the embeddings file.
        :param dataset_file: path to a pre-labeled dataset file.
        :param plot_roc: boolean, plot ROC figure.
        :param split: float, data split fraction i.e. train | test split (default: 80% | 20%)
        :param detailed: boolean, output classifiers' parameters info i.e. name, parameters' value, .. etc.
        """

        plt.clf()
        self.test_indx = processData.get_test_seq()
        self.split = split
        self.isTest = is_Test

        #read dataset
        train, test = self.read_data(dataset)
        self.X_train, self.X_test = train[:,:-1], test[:,:-1]
       
        self.y_train = train[:,-1:]
        self.y_test = test[:,-1:]
        self.accuracies = {}
        self.cmatrix = {}
        plt.Figure()

        if self.isTest:
            # RUN classifiers
            for c in processData.theClassifiersLevel2:
                self.cmatrix[c.__class__.__name__] = self.predict(c, detailed, plot_roc)
        else:
            # RUN classifiers
            for c in processData.theClassifiersLevel2:
                self.cmatrix[c.__class__.__name__] = self.classify(c, detailed, plot_roc)

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
            rstring = '{}\n-----------mmmmmmmmmmmmmmmm-----------\n{}'
            cres = rstring.format(k, v) 
            print(cres)
            allresults += '\n' + cres

        print('Overall avg F1 test {:.2f}%'.format((avg_f1 / len(self.accuracies)) * 100))
        info("DONE!")

    def read_data(self, dataset_in):
        dataset = dataset_in
        if self.isTest:
            # it is a test, no need for split.
            # run prediction on the whole test dataset.
            return dataset, dataset
        else:
            # split train/test
            #train_df, test_df = train_test_split(dataset, train_size=self.split,random_state=42)
            train_df, test_df = train_test_split(dataset, train_size=self.split,random_state=2812)
            return train_df, test_df

    def classify(self, classifier=None, info_=False, plot_roc=False):
        classifier_name = classifier.__class__.__name__
        print('classifier_name',classifier_name)
        if info_:
            info('fitting data ...')
            info('\n\ncreated \n\n{}'.format(classifier))

        classifier.fit(self.X_train, self.y_train)
        # now you can save it to a file
        with open('Results/L2'+classifier_name, 'wb') as f:
            pickle.dump(classifier, f)

        pscore = classifier.score(self.X_test, self.y_test)
        
        if info_:
            info('\n\t{}() ACCURACY: {}\n'.format(classifier_name, pscore))

        # F1 score
        y_pred = classifier.predict(self.X_test)
        f1_score = metrics.f1_score(self.y_test, y_pred,zero_division=0)

        # macro accuracy (macro average)
        macc = metrics.f1_score(self.y_test, y_pred, pos_label=None, average='macro',zero_division=0)

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

    def predict(self, classifier=None, info_=False, plot_roc=False):
        classifier_name = classifier.__class__.__name__
        print('classifier_name',classifier_name)
        # load the classifier
        with open('Results/L2'+classifier_name, 'rb') as f:
            classifier = pickle.load(f)

        print(len(self.X_test),len(self.y_test))
        pscore = classifier.score(self.X_test, self.y_test)
        # pscore = classifier.score(self.X_train, self.y_train)
        info('\n\n\t{}() ACC.: {}\n'.format(classifier_name, pscore))

        # F1 score
        y_pred = classifier.predict(self.X_test)
        f1_score = metrics.f1_score(self.y_test, y_pred,zero_division=0)

        # macro accuracy (macro average)
        macc = metrics.f1_score(self.y_test, y_pred, pos_label=None, average='macro',zero_division=0)
        confusion_matrix= metrics.confusion_matrix(self.y_test,  y_pred)
        print('...............confusion................')
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
            classifier_probas = estimator.decision_function(self.X_test)
        except AttributeError:
            classifier_probas = estimator.predict_proba(self.X_test)[:, 1]

        false_positive_r, true_positive_r, thresholds = metrics.roc_curve(self.y_test, classifier_probas)
        roc_auc = metrics.auc(false_positive_r, true_positive_r)

        label = '{:.1f}% neg:{} pos:{} {}'.format(roc_auc * 100, neg, pos, estimator_name)
        roc_path = 'Results/ROC_{}.png'.format(self.test_indx)
        plt.plot(false_positive_r, true_positive_r, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('ROC score(s)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right', prop={'size': 10})
        plt.savefig(roc_path, dpi=600, bbox_inches='tight')
        plt.grid()
        

