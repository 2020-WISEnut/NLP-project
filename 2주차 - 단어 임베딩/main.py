# -*- coding: utf-8 -*-

from numpy.lib.function_base import vectorize
from numpy.lib.npyio import save
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
        model_arg = config["model_arg"]
        save_option = config["save_option"]
        save_path = config["save_path"]
        
        # load data and preprocess
        testset, vocab = self.load_testset(load_path)
        datasets = self.load_data(load_path, vocab, preprocess_option)

        # get word vectors
        word_vectors = self.get_word_vectors(datasets, vectorizer_type, model_arg)
        # save word vectors
        if save_option:
            file_name = "size300_win8_cnt10.wv"
            self.save_word_vectors(word_vectors, save_path, file_name, vectorizer_type)

        # get lists of word similarities
        # 단어 유사도 리스트
        answer_list, pred_list = self.get_word_similarity(word_vectors, testset, vectorizer_type)

        # get spearman and pearson correlation
        spearman, pearson = self.get_correlation(answer_list, pred_list)
        print('spearman: %.5f, pearson: %.5f' % (spearman, pearson))

    def get_config(self):
        try:
            with open("config.yaml", "r", encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            print("config.yaml not found")
            sys.exit()

        return config

    def load_testset(self, load_path):
        testset_path = os.path.join(load_path, 'kor_ws353.csv')
        testset = []
        vocab = set()
        try:
            with open(testset_path, 'r', encoding='utf-8-sig') as testfile:
                for pair in testfile:
                    temp = pair.strip().split(',')
                    testset.append(temp)   
                    vocab.add(temp[0])   
                    vocab.add(temp[1])
        except IOError:
            print("fail to open file")
            sys.exit()

        return testset, vocab

    def load_data(self, load_path, vocab, preprocess_option):
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
        try:
            with open(os.path.join(load_path, corpus_file_name), 'r', encoding='utf-8') as f:
                for line in f:
                    dataset = self.preprocess(line, preprocess_option, weird_words)
                    if self.has_vocab(dataset, vocab):
                        datasets.append(dataset)
        except IOError:
            print("failed to open file")
            sys.exit()

        return datasets

    def preprocess(self, text, preprocess_option, weird_words):
        if preprocess_option in [1, 3]:
            pat = r"[^가-힣A-Za-z0-9]+"
        elif preprocess_option in [2, 4]:
            pat = r"[^가-힣A-Za-z]+"
        else:
            print("Preprocessing option not valid.")
            sys.exit()

        dataset = re.sub(pat, " ", text).strip().split()
        for i in range(len(dataset)-1):
            if dataset[i] in weird_words and dataset[i+1] == weird_words[dataset[i]]:
                dataset[i], dataset[i+1] = (dataset[i] + dataset[i+1]), ""
        
        text = ' '.join(dataset)
        if preprocess_option in [3, 4]:
            text = self.remove_stopwords(text)

        return text.split()

    def remove_stopwords(self, text: str):
        with open("stopwords.txt", encoding="utf-8") as stopwords:
            for stopword in stopwords:
                text = text.replace(stopword.strip(), "")
        
        return text

    def has_vocab(self, dataset: str, vocab: set):
        for word in vocab:
            if word in dataset:
                return True

        return False

    def get_word_vectors(self, datasets, vectorizer_type, model_arg):
        vectorizer = Vectorizer(vectorizer_type, model_arg)
        word_vectors = vectorizer.vectorize(datasets)

        return word_vectors

    def save_word_vectors(self, word_vectors, save_path, file_name, vectorizer_type):
        if vectorizer_type == 1:
            word_vectors.save_word2vec_format(os.path.join(save_path, file_name))

    def get_word_similarity(self, word_vectors, testset, vectorizer_type):
        answer_list, pred_list = [], []

        if vectorizer_type == 1:
            for pair in testset:
                w1, w2, sim = pair.strip().split(',')
                try:
                    pred = word_vectors.similarity(w1, w2)
                    answer_list.append(float(sim))
                    pred_list.append(pred)
                except KeyError as e:
                    # 단어 임베딩에 포함되지 않은 단어들
                    print(e)
        elif self.vectorizer_type == 2:
            ############ Fasttext ############
            pass
        elif self.vectorizer_type == 3:
            ############# GloVe ##############
            pass
        elif self.vectorizer_type == 4:
            ############# Swivel #############
            pass
        else:
            print("vectorizer type not valid.")
            sys.exit()      

        return answer_list, pred_list

    def get_correlation(self, answer_list, pred_list):
        spearman, _ = stats.spearmanr(answer_list, pred_list)
        pearson, _ = stats.pearsonr(answer_list, pred_list)

        return spearman, pearson


class Vectorizer:
    def __init__(self, vectorizer_type, model_arg):
        self.vectorizer_type = vectorizer_type
        self.model_arg = model_arg

    def vectorize(self, datasets):
        if self.vectorizer_type == 1:
            model = Word2Vec(datasets, **self.model_arg)
            word_vectors = model.wv
        elif self.vectorizer_type == 2:
            ############ Fasttext ############
            pass
        elif self.vectorizer_type == 3:
            ############# GloVe ##############
            pass
        elif self.vectorizer_type == 4:
            ############# Swivel #############
            pass
        else:
            print("vectorizer type not valid.")
            sys.exit()
        
        return word_vectors


if __name__ == "__main__":
    wv_corr = WordVectorCorrelation()
    wv_corr.run()
