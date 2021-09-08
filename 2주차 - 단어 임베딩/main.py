# -*- coding: utf-8 -*-

from numpy.lib.function_base import vectorize
import yaml
import re
#from gensim.models import Word2Vec
import os
from scipy import stats
from matplotlib import pyplot as plt


class WordVectorCorrelation:
    def __init__(self):
        pass

    def run(self):
        # read config.yaml and set config
        config = self.get_config()
        load_path = config["load_path"]
        preprocess_option = config["preprocess_option"]
        vectorizer_type = config["vectorizer_type"]
        save_path = config["save_path"]

        # load data and preprocess
        datasets = self.load_data(load_path)

        # get word vectors
        word_vectors = self.get_word_vectors(datasets, vectorizer_type)

        # save word vectors
        file_name = "size300_win8_cnt10.wv"
        self.save_word_vectors(word_vectors, save_path, file_name, vectorizer_type)

        # get lists of word similarities
        # 단어 유사도 리스트
        answer_list, pred_list = self.get_word_similarity(word_vectors, load_path)

        # get spearman and pearson correlation
        spearman, pearson = self.get_correlation(answer_list, pred_list)
        print('spearman: %.5f, pearson: %.5f' % (spearman, pearson))

    def get_config(self):
        pass

    def load_data(self, load_path, preprocess_option):
        datasets = []
        # load data
        corpus_file_name = "wiki_ko_mecab.txt"
        pass

        self.preprocess(preprocess_option)

        return datasets

    def preprocess(self, preprocess_option):
        pass

    def get_word_vectors(self, datasets, vectorizer_type):
        vectorizer = Vectorizer(vectorizer_type)
        word_vectors = vectorizer.vectorize(datasets)

        return word_vectors

    def save_word_vectors(self, word_vectors, save_path, file_name, vectorizer_type):
        pass

    def get_word_similarity(self, word_vectors, load_path):
        answer_list, pred_list = [], []

        return answer_list, pred_list

    def get_correlation(self, answer_list, pred_list):
        pass


class Vectorizer:
    def __init__(self, vectorizer_type):
        self.vectorizer_type = vectorizer_type

    def vectorize(self):
        pass


if __name__ == "__main__":
    wv_corr = WordVectorCorrelation()
    wv_corr.run()