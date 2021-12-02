# coding: utf-8
    
"""
runHAteSpeech-test.py is a file to predict Hate Speech on Arabic test dataset 
    (Levantine dataset from twitter) using Word Embeddings.
    We use voting on several classiefiers.
        Hard voting.
        Soft voting.
        Our own ensembling.
@author: Medyan
Date: Dec. 2020
"""
import pickle
import argparse
import HateSpeech
import sys

if __name__ == "__main__":
    stdoutorigin = sys.stdout
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", help="path a pre-trained vectors model.")
    parser.add_argument("--datatype", help="data type")
    parser.add_argument("--dataset", help="path a labeled (0/1) sentiment dataset.")
    parser.add_argument("--voting_type", help="use voting model or conventional models.")
    parser.add_argument("--isTest", help="train the model or apply the model on test dataset.")
    parser.add_argument("--plot", help="print ROC curve or not")
    
    args = parser.parse_args()
    vec = args.vectors
    '''
    voting_type values:
        0- one layer model; no voting or ensembling. 
        1- two layers model; soft/hard voting model. 
        2- two layers model; special ensembling model. 
    '''
    # vectors file and dataset pathes
    datatype = args.datatype if args.datatype else "E"
    embeddings_path = args.vectors if args.vectors else "arabic.bin"
    # dataset_path = args.dataset if args.dataset else "thesis-trainE.csv"
    # dataset_path = args.dataset if args.dataset else "thesis-testsetB.csv"
    # dataset_path = args.dataset if args.dataset else "CleanL-HSAB-AbusHateTrain.csv"
    # dataset_path = args.dataset if args.dataset else "CleanL-OSACT2020-sharedTask-train.csv"
    dataset_path = args.dataset if args.dataset else "offenseval-ar-training-v2.csv"
    voting_type = args.voting_type if args.voting_type else 0
    isTest = args.isTest if args.isTest else True
    plot = args.plot if args.plot else True
    #  split  dataset into two halfs for each layer
    split = 0.5 if voting_type == 2 else 0.8

    # run
    outputFilename = "Results/no-augment-{}{}{}{}.txt".format(dataset_path, 
                                                              voting_type, 
                                                              datatype, 
                                                              462 
                                                              )
    # MAXIMUM PROBABILITY   262 - 266
    # SUM PROBABILITY       267 - 268
    # print('outputFilename: {} '.format(outputFilename))
    sys.stdout = open(outputFilename , "w")
    medyan = HateSpeech.HateSpeechDetection(embeddings_path, 
                                            datatype,
                                            dataset_path, 
                                            split=split,
                                            voting_type=voting_type,
                                            runTest=isTest,
                                            plot_roc=plot)
    sys.stdout.close()
    sys.stdout = stdoutorigin
