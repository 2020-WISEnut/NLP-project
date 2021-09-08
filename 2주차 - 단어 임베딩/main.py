# -*- coding: utf-8 -*-

from numpy.lib.function_base import vectorize
import yaml
import re
from gensim.models import Word2Vec
import os
import sys
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
        model_arg = config["model_arg"]

        # load data and preprocess
        datasets = self.load_data(load_path)

        # get word vectors
        word_vectors = self.get_word_vectors(datasets, vectorizer_type, model_arg)
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
        ##################### TO DO #####################
        pass

    def load_data(self, load_path, preprocess_option):
        datasets = []
        # load data
        corpus_file_name = "wiki_ko_mecab.txt"
        
        weird_words = {"똑똑": "한", 
                       "씨": "디",
                       "고양이": "과",
                       "육식": "동물",
                       "정신": "의학",
                       "수집": "품",
                       "독립": "체",
                       "인종": "차별",
                       "유사": "도",
                       "근접": "성",
                       "랍": "스터",
                       "중요": "성",
                       "멍청": "한"}
        
        with open(os.path.join(load_path, corpus_file_name), 'r', encoding='utf-8') as f:
            for line in f:
                dataset = self.preprocess(line, preprocess_option, weird_words)
                datasets.append(dataset)
        
        return datasets

    def preprocess(self, text, preprocess_option, weird_words):
        if preprocess_option in [1, 3]:
            pat = r"[^가-힣A-Za-z]+"
        elif preprocess_option in [2, 4]:
            pat = r"[^가-힣A-Za-z0-9]+"
        else:
            print("Preprocessing option not valid.")
            sys.exit()

        dataset = re.sub(pat, " ", text).strip().split()
        for i in range(len(dataset)-1):
            if dataset[i] in weird_words and dataset[i+1] == weird_words[dataset[i]]:
                dataset[i], dataset[i+1] = (dataset[i] + dataset[i+1]), ""
        
        return ' '.join(dataset).split()

    def get_word_vectors(self, datasets, vectorizer_type, model_arg):
        vectorizer = Vectorizer(vectorizer_type, model_arg)
        word_vectors = vectorizer.vectorize(datasets)

        return word_vectors

    def save_word_vectors(self, word_vectors, save_path, file_name, vectorizer_type):
        ##################### TO DO #####################
        pass

    def get_word_similarity(self, word_vectors, load_path):
        answer_list, pred_list = [], []
        ##################### TO DO #####################

        return answer_list, pred_list

    def get_correlation(self, answer_list, pred_list):
        ##################### TO DO #####################
        pass


class Vectorizer:
    def __init__(self, vectorizer_type, model_arg):
        self.vectorizer_type = vectorizer_type
        self.model_arg = model_arg

    def vectorize(self, datasets):
        if self.vectorizer_type == 1:
            model = Word2Vec(datasets, **self.model_arg)
            word_vectors = model.wv
        
        return word_vectors


if __name__ == "__main__":
    wv_corr = WordVectorCorrelation()
    wv_corr.run()
