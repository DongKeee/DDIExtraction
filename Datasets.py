# -*- coding:UTF-8 -*-
from classEntity import Sentence, RelationPair
from dataProcess import load_data,Initial
import numpy as np
import math
import copy



initial = Initial(all_data_path="data/all/")

class Datasets(object):
    def __init__(self, filename="data//test"):
        print("start to load data from path", filename)
        # self.intial=Initial(all_data_path=filename)
        self.filename = filename
        self.features = list()
        self.sentences,self.filenameList = load_data(filename)
        self.positive = 0
        self.other = 0
        self.mechanism = 0
        self.int = 0
        self.advise = 0
        self.effect = 0
        self.lengthM=0
        self.get_features()

    def get_features(self):
        for sentence in self.sentences:
            f = Feature(sen=sentence)
            for instance in f.instances:
                self.features.append(instance)
                self.lengthM=max(len(f.words),self.lengthM)
            self.positive +=f.positive
            self.other +=f.other
            self.mechanism +=f.mechanism
            self.int +=f.int
            self.advise +=f.advise
            self.effect +=f.effect

class Feature(object):
    def __init__(self, sen=Sentence(), padding=1):
        self.sen = sen
        self.words = self.sen.new_context.split("@@")#用@@将分词后的内容获取出来
        self.instances = list()
        self.max_length = 150#文本中词语个数最大值
        self.positive = 0
        self.other = 0
        self.mechanism = 0
        self.int = 0
        self.advise = 0
        self.effect = 0
        self.features()
        self.padding = padding

    def features(self):
        for relation in self.sen.relation_list:
            instance = dict()
            e1_position = relation.e1_position
            e2_position = relation.e2_position
            padding_words = self.words[e1_position:e2_position + 1]
            padding_words[0] = "DRUG1"
            padding_words[-1] = "DRUG2"
            drug = set()
            for entity in self.sen.entity_list:
                drug.add(entity.text)
            for i in range(len(padding_words)):
                if padding_words[i] in drug:
                    padding_words[i] = 'DRUG0'#对填充部分进行药物掩饰

            # generate all the words
            sentence_words = copy.deepcopy(self.words)
            for i in range(len(sentence_words)):
                if sentence_words[i] in drug:
                    sentence_words[i] = "DRUG0"
            sentence_words[relation.e1_position] = "DRUG1"
            sentence_words[relation.e2_position] = "DRUG2"

            # padding the </s>
            left = (self.max_length - len(sentence_words)) // 2#左边填充的个数
            right = self.max_length - left - len(sentence_words)#右边填充的个数
            sentence_words_padding = ["</s>"] * left + sentence_words + ["</s>"] * right
            assert len(sentence_words_padding) == self.max_length

            all_sequence = [initial.word2index[word] for word in sentence_words_padding]
            sequence = [initial.word2index[word] for word in padding_words]

            instance['entityNum'] = len(self.sen.entity_list)
            instance['padding_words'] = padding_words

            instance['orilText'] = ' '.join(copy.deepcopy(self.words))

            # ------->整个句子的单词
            instance['word_sequence'] = sentence_words
            # ------->第一个实体和第二个实体之间单词的index
            instance['sequence'] = sequence
            # ------->整个句子单词的index
            instance['all_sequence'] = all_sequence
            # ------->第一个实体在句子中的位置
            instance['e1_pos'] = relation.e1_position+left
            # ------->第二个实体在句子中的位置
            instance['e2_pos'] = relation.e2_position+left
            # ------->最短路径，以及对应的index
            # instance['sdp'] = sdp
            # ------->关系对应的type，总共有四种
            instance['type'] = relation.type
            # ------->是否存在DDI之间的关系
            instance['ddi'] = relation.ddi
            # ------->type对应的class标号，numpy表示
            instance['class'] = np.array([initial.label[relation.type]])
            # ------->type对应的class标号
            instance['label'] = initial.label[relation.type]
            # ------->对应的二分类时候的标签
            instance['binary'] = 0 if relation.type is "other" else 1
            # ------->对应的二分类时候的标签，numpy表示
            instance['binary_class'] = np.array([instance['binary']])
            # print sequence
            # ------->instance['negative'], 决定是否提前过滤掉该关系
            instance['negative'] = False
            
            instance['negative'] = self.filter(instance, relation)
            instance['relation'] = relation
            instance['context'] = self.sen.new_context
            self.instances.append(instance)
            if instance['negative'] == False:
                if relation.type == "mechanism":
                    self.positive += 1
                    self.mechanism += 1
                elif relation.type == "int":
                    self.int += 1
                    self.positive += 1
                elif relation.type == "advise":
                    self.positive += 1
                    self.advise += 1
                elif relation.type == "effect":
                    self.effect += 1
                    self.positive += 1
                else:
                    self.other += 1


            # print(instance)
    # to decide whether the two entities are illegal
    def filter(self, instance, relation=RelationPair()):
        e1_name = str(relation.e1_name).lower()
        e2_name = str(relation.e2_name).lower()

        return self.filter_1(e1_name, e2_name) \
               or self.filter_2(e1_name, e2_name) \
               or self.filter_3(relation, instance) \
               or self.filter_4(instance=instance)

    # 判断名称是否一样
    def filter_1(self, e1_name, e2_name):
        return e1_name == e2_name

    # 判断一个名称是否是另一个名称的缩写
    def filter_2(self, e1_name, e2_name):
        if len(str(e1_name).split(" ")) > 1:
            if len(str(e2_name).split(" ")) == 1:
                split_words = str(e1_name).split(" ")
                line = "".join([word[0] for word in split_words if str(word).rstrip() != ""])
                return line == e2_name

        if len(str(e2_name).split(" ")) > 1:
            if len(str(e1_name).split(" ")) == 1:
                split_words = str(e2_name).split(" ")
                # print "split words", split_words, "\t", e1_name, e2_name
                line = "".join([word[0] for word in split_words if str(word).rstrip() != ""])
                return line == e1_name

    # 判断 A [and, or, ,, (,] B 的情况
    # 判断 A , or 这种情况
    def filter_3(self, relation=RelationPair(), instance=dict()):
        e1_pos = relation.e1_position
        e2_pos = relation.e2_position
        if math.fabs(e2_pos - e1_pos) == 1:
            return True

        if math.fabs(e2_pos - e1_pos) == 2:
            between = str(self.words[min(e1_pos, e2_pos) + 1]).lower()
            # if between == "and" \
            # or between == "or" \
            #         or between == "," \
            #         or between == "(" \
            #         or between == "-":
            if between == "or" \
                    or between == "," \
                    or between == "(" \
                    or between == "-" \
                    or between == "and":
                return True
                # if between == "and" and e1_pos - 1 >= 0:
                #     word = str(instance['word_sequence'][e1_pos - 1]).lower()
                #     if word not in ["of", "between", "with"]:
                #         return True

        if math.fabs(e2_pos - e1_pos) == 3:
            minvalue = min(e1_pos, e2_pos)
            word = str(" ".join(self.words[minvalue + 1: minvalue + 3])).lower()
            if word == ", or" or word == "such as":
                return True

    # filter 掉并列的结构，这个很重要
    # a,b,c, and d
    def filter_4(self, instance=None):
        except_words = [",", 'drug0', 'or', '(', '[', ')', ']', "and"]
        flags = False
        if not instance:
            instance = dict()
        e1_pos = instance['e1_pos']
        e2_pos = instance['e2_pos']
        sequence = instance['all_sequence']
        # print sequence
        for i in range(e1_pos + 1, e2_pos):
            # print('aaa')
            word = str(sequence[i]).lower()
            if word not in except_words:
                return False
            else:
                if word == "and":
                    flags = True
        if flags is True:
            if e2_pos - e1_pos <= 4:
                return False
        return True
