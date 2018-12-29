# -*- coding:UTF-8 -*-
# from __builtin__ import object
from xml.etree import ElementTree
# import nltk
# nltk.download()
# from nltk.tokenize import WordPunctTokenizer
# from nltk.tokenize import word_tokenize,sent_tokenize


import re
class Document(object):
    def __init__(self, filename):
        self.filename = filename
        self.input_xml = open(filename).read()
        self.root = ElementTree.fromstring(self.input_xml)
        self.sentence_list = list()
        self.text = []
        self.generate_data_from_xml()


    def generate_data_from_xml(self):
        for i in range(len(self.root)):
            sentence = self.root[i]
            sen = Sentence()
            sen.original_context = sentence.attrib['text']
            sen.new_context = sentence.attrib['new_text']
            sen.id = sentence.attrib['id']
            entity_list = list()
            relation_list = list()

            for j in range(len(sentence)):#处理句子中的实体对象
                node = sentence[j]
                if node.tag == "entity":
                    entity = Entity()
                    entity.id = node.attrib['id']
                    entity.position = -1 if node.attrib['position'] == "not_a_number" else int(node.attrib['position'])
                    entity.text = node.attrib['text']
                    entity_list.append(entity)
                    # entity.smart_print()
                else:
                    if "e1_pos" in node.attrib:
                        relation = RelationPair()
                        relation.e1_id = node.attrib['e1']
                        relation.e2_id = node.attrib['e2']
                        relation.e1_position = int(node.attrib['e1_pos'])
                        relation.e2_position = int(node.attrib['e2_pos'])
                        relation.e1_name = node.attrib['e1_name']
                        relation.e2_name = node.attrib['e2_name']
                        relation.path = node.attrib['path']
                        # relation.sdp = node.text
                        relation.id=node.attrib['id']
                        relation.type = node.attrib['type'] if 'type' in node.attrib else "other"
                        relation.flags = True if node.attrib['flags'] == "True" else False
                        relation.ddi = True if node.attrib['ddi'] == "True" else False
                        relation_list.append(relation)
            sen.entity_list = entity_list
            sen.relation_list = relation_list
            self.sentence_list.append(sen)


class Sentence(object):
    def __init__(self):
        # 初始的文本内容
        self.original_context = None
        # split with @@
        self.new_context = None
        # 实体的链表
        self.entity_list = list()
        # 关系链表
        self.relation_list = list()
        # id
        self.id = None


class Entity(object):
    def __init__(self):
        self.id = None
        self.text = None
        self.type = None
        self.charOffset=None
        self.position = None

    def smart_print(self):
        print("########################")
        print("id = ", self.id, "\n", \
            "text = ", self.text, "\n", \
            "type = ", self.type, "\n", \
            "position = ", self.position)

        print("########################")


class RelationPair(object):
    def __init__(self):
        self.e1_name = None
        self.e2_name = None
        self.e1_position = None
        self.e2_position = None
        self.e1_id = None
        self.e2_id = None
        self.id = None
        # self.flags = None
        self.type = None
        # self.sdp = None
        # self.path = None
        self.ddi = None

    def smart_print(self):
        print("########################")
        print("e1_name = ", self.e1_name, "\n", \
            "e2_name = ", self.e2_name, "\n", \
            "e1_position = ", self.e1_position, "\n", \
            "e2_position = ", self.e2_position, "\n", \
            "e1_id = ", self.e1_id, "\n", \
            "e2_id = ", self.e2_id, "\n", \
            "id = ", self.id, "\n", \
            # "flags = ", self.flags, "\n", \
            "type = ", self.type, "\n", \
            # "sdp = ", self.sdp, "\n", \
            "ddi = ", self.ddi, "\n", \
            # "path = ", self.path
              )
        print("########################")




