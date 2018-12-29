# -*- coding:UTF-8 -*-
from classEntity import Document
# from progressbar import *
import numpy as np
import os


def load_data(path=""):
    # time.sleep(0.5)
    print("start to load data from path----->", path)
    # time.sleep(0.5)
    file_list = os.listdir(path)
    sentences = list()
    other,positive,mechanism,int,advise,effect=0,0,0,0,0,0
    fileNameList=[]
    for i in range(len(file_list)):
        filename = file_list[i]
        current_path = os.path.join(path, filename)#将多个路径组合后返回
        document = Document(filename=current_path)
        for sentence in document.sentence_list:
            fileNameList.append(current_path)
            sentences.append(sentence)
    return sentences,fileNameList
    # return alltext

class Initial(object):
    def __init__(self,
                 all_data_path,
                 vector_length=200,#词向量的维度
                 pre_trained_embedding="data//wiki_pubmed"):
        self.all_data_path = all_data_path
        self.pre_trained_embedding = pre_trained_embedding

        # word2index:每个词及其对应出现的编号 and index2word：每个编号对应的词
        self.word2index = dict()
        self.index2word = dict()
        self.word2index['</s>'] = 0
        self.word2index['DRUG1'] = 1
        self.word2index['DRUG2'] = 2
        self.word2index['DRUG0'] = 3
        self.index()

        # init the word vectors
        self.vector_length = vector_length
        filename='data//wiki_pubmed'
        self.word_dict = self.init_word_embedding(filename)

        # init the label初始化标签的编号
        self.label = dict()
        self.label['int'] = 0
        self.label['advise'] = 1
        self.label['effect'] = 2
        self.label['mechanism'] = 3
        self.label['other'] = 4

    def index(self):
        print("start index the data in ", self.all_data_path)
        sentences,filema = load_data(path=self.all_data_path)
        current_index = 4
        for sentence in sentences:
            words = str(sentence.new_context).strip("\r").strip("\n").rstrip().split("@@")
            for word in words:
                if word not in self.word2index:
                    self.word2index[word] = current_index
                    current_index += 1
        for word in self.word2index:
            self.index2word[self.word2index[word]] = word

    def init_word_embedding(self, filename):
        # init the pre_trained_embedding from text
        print("start to load the pre trained word embedding from", filename)
        Word = np.random.uniform(low=-0.1, high=0.1, size=(len(self.word2index), self.vector_length))#按照【词语个数*词向量维度】尺寸随机初始化词向量矩阵，使得未训练到的词语用随机初始化的向量值
        openfile = open(filename)
        word_2_vector = dict()
        for line in openfile:
            words = str(line).strip("\r\n").split(" ")
            word_2_vector[words[0]] = [float(words[i]) for i in range(1, len(words))]#记录每个词语的词向量
        openfile.close()

        word_2_vector['</s>'] = [0.0] * self.vector_length#用0填充
        # init the self.Words
        word_in_pretrained = 0
        for i in range(len(self.word2index)):
            word = self.index2word[i]
            if word in word_2_vector:
                Word[i] = word_2_vector[word]
                word_in_pretrained += 1
        print("所有样本中所有词语的个数all the word size is --->", len(self.word2index))
        print("预训练好的词向量个数words in pre-trained word embedding is ---# >", word_in_pretrained)
        return Word







    #word vector representation

    # print('测试集合DDI数量',len(sen_test))
    # for i in range(len(sen_test)):
    #     print(sen_test[i])

    # sen_train = load_data('data/train')
