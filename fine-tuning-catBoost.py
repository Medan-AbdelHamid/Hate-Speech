# coding: utf-8

"""
asa.py is a simple (Arabic) Sentiment Analysis using Word Embeddings.
Author: Aziz Alto
Date: Aug. 2016
"""
from catboost import CatBoostClassifier
from collections import Counter
import pickle
import sqlite3
import argparse
from logging import info, basicConfig, INFO
# -- 3rd party -- #
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize
# -- classifiers -- #
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
LOG_HEAD = '[%(asctime)s] %(levelname)s: %(message)s'
basicConfig(format=LOG_HEAD, level=INFO)


def get_test_seq():
    conn = sqlite3.connect('fdata-2020.db')
    sql = 'SELECT seq FROM SQLITE_SEQUENCE where name=\'TESTS\''
    cur = conn.cursor()
    cur.execute(sql)
    last = cur.fetchone()[0] + 1
    conn.close()
    return last

def add_test_to_database(results, notes):
    conn = sqlite3.connect('fdata-2020.db')
    sql = "INSERT INTO TESTS (RESULTS, NOTES)  VALUES (?, ?)"
    args = (results, notes,)
    conn.execute(sql, args)
    conn.commit()

def get_vec(n_model,dim, token):
    vec = np.zeros(dim)
    if token not in n_model.wv:
        _count = 0
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
    def __init__(self, embeddings_file=None, dataset_file=None, plot_roc=False, split=0.80 , detailed=False):
        """
        :param embeddings_file: path to the embeddings file.
        :param dataset_file: path to a labeled dataset file.
        :param plot_roc: boolean, plot ROC figure.
        :param split: float, data split fraction i.e. train | test split (default: 90% | 10%)
        :param detailed: boolean, output classifiers' parameters info i.e. name, parameters' value, .. etc.
        """

        self.test_indx = get_test_seq()
        self.dataset_file = dataset_file
        self.split = split

        self.embeddings, self.dimension = self.load_vectors(embeddings_file)
        #input('embedding')
        #read dataset
        train, test = self.read_data(self.dataset_file)
        train_txt, test_txt = train['FULLTEXT'], test['FULLTEXT']
       
        self.y_train = train['ANOMALY']
        self.y_test = test['ANOMALY']
        # -- dataset preprocessing -- #
        train_tokens = self.tokenize_data(train_txt, 'training')
        test_tokens = self.tokenize_data(test_txt, 'testing')
        # -- vectorize training/testing data -- #
        train_vectors = self.average_feature_vectors(train_tokens, 'training')
        test_vectors = self.average_feature_vectors(test_tokens, 'testing')
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
        self.X_train,self.X_test,self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        '''
        print('Original dataset shape %s' % Counter(self.y_train))
        '''
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
        print('Resampled dataset shape %s' % Counter(self.y_train))
        ''' 
        
        param_test1 = {
            'learning_rate': [0.001, 0.005, 0.01], #best value: 5e-3
            'iterations': [1000, 2000, 5000], #best value: 2000
            'depth':[6,7,8], #best value 7
            'l2_leaf_reg':[5, 6, 7], # 6
            'border_count':[5,10,20,32,50,100,200],# 10
            'ctr_border_count':[5,10,20,50,100,200], #200
            'thread_count':[1, 4] # 4
          }
        gsearch1 = GridSearchCV(estimator = CatBoostClassifier(iterations=2000, learning_rate=0.005, depth=7, l2_leaf_reg=6, border_count=32, thread_count=1), 
                                param_grid = param_test1, 
                                scoring='roc_auc',
                                n_jobs=4,
                                iid=False, 
                                cv=5)
        gsearch1.fit(self.X_train, self.y_train)
        gsearch1.best_params_, gsearch1.best_score_
        print(gsearch1.best_score_)
        print(gsearch1.best_params_)
        #input('ok')
        info('Done loading and vectorizing data.')
        info("--- Sentiment CLASSIFIERS ---")
        info("fitting ... ")


        #instantiate model and train

        avg_f1 = 0
        allresults = 'Results: '
        info('results ...')
        for k, v in self.accuracies.items():
            rstring = '\tMacAvg. {:.2f}% F1. {:.2f}% P. {:.2f} R. {:.2f} Acc. {:.2f}%: {}'
            cres = rstring.format(v[0] * 100, v[1] * 100, v[2] * 100, v[3] * 100,v[4] * 100, k) 
            print(cres)
            allresults += '\n' + cres
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
        w2v_model.init_sims(replace=True)  # to save memory
        vocab, vector_dim = w2v_model.wv.syn0.shape
        #print('model.corpus_count',w2v_model.corpus_count)
        #print('model.corpus_total_words',w2v_model.corpus_total_words)
        #print('len(model.wv.vocab)',len(w2v_model.wv.vocab))
            # similarity between two different words
        #print("Similarity between 'المرض' and 'الالم'",w2v_model.wv.similarity(w1="المرض", w2="الالم"))
        #input('load')
        return w2v_model, vector_dim

    def read_data(self, dataset_in):
        dataset = pd.read_csv(dataset_in)
        
        # split train/test
        #train_df, test_df = train_test_split(dataset, train_size=self.split,random_state=42)
        train_df, test_df = train_test_split(dataset, train_size=self.split,random_state=42)
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
        # print('+++++++++++++++++++++++++++++++++++++++')
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

    def classify(self, classifier=None, idx=0, info_=False, plot_roc=False):

        classifier_name = classifier.__class__.__name__ + str(idx)
        print('classifier_name',classifier_name)
        if info_:
            info('fitting data ...')
            info('\n\ncreated \n\n{}'.format(classifier))

        classifier.fit(self.X_train, self.y_train)
        # now you can save it to a file
        with open('Results/'+classifier_name, 'wb') as f:
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
    dataset_path = args.dataset if args.dataset else "thesis-trainset.csv"
    # run
    medyan = ArSentiment(embeddings_path, dataset_path, plot_roc=True)
