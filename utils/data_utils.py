import codecs
import json
from os.path import join
import pickle
import os
import numpy.random as random
import math
import utils.string_utils as string_utils

'''pub中的格式'''
# ''' "id":"","title":"","abstract":"","keywords": ["",""],
#     "authors": [{"name":"","org":""},{"name":"","org":""}],"venue": "","year": 2009
# '''
# pub_dict = ['id','title','abstract','keywords','authors','venue','year']

def load_json(rfdir='../data/', rfname='train_pub.json'):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)



def dump_json(obj, wfpath, wfname, indent=None):
    # with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
    #     json.dump(obj, wf, ensure_ascii=False, indent=indent)
    json_str = json.dumps(obj, indent=indent)
    with open(join(wfpath, wfname), 'w') as json_file:
        json_file.write(json_str)

def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)

def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(s):
    return pickle.loads(s)


pub_string_item = ['title','abstract','venue']
def pre_data(author_dict,pub_dict): # 预处理字符串数据，去掉字符串中的停用词，将名字统一成小写
    author_new = dict() # 对作者名字进行预处理过的数据
    for name in author_dict:
        new_name = string_utils.clean_name_author(name)
        author_new[new_name] = author_dict[name]
    #
    # # 把pub中的字符串都进行预处理
    for id in pub_dict:
        for item in pub_string_item:
            pub_dict[id][item] = string_utils.clean_sentence(pub_dict[id][item])
        authors=[]
        for author in pub_dict[id]["authors"]:
            authors.append(
                {'name':string_utils.clean_name(author['name']),
                 'org':string_utils.clean_sentence(author['org'])}
            )
        pub_dict[id]["authors"] =authors

    dump_json(author_new, wfpath='../data/', wfname='train_author_new.json', indent=4)
    dump_json(pub_dict, wfpath='../data/', wfname='train_pub_new.json', indent=4)
    return author_new,pub_dict


def divide_data(author_dict): # 划分数据集
    total_pub =0
    for author in author_dict:
        for pub in author_dict[author]:
            total_pub = total_pub+len(pub)
    train_pub = total_pub*0.8 # 80%作为训练集

    keys = list(author_dict.keys())
    random.shuffle(keys) # 为数据随机排列

    train_author = dict()
    test_author = dict()
    train_num= 0
    for key in keys:
        if(train_num<train_pub):
            train_author[key] = author_dict[key]
            for pub in author_dict[key]:
                train_num = train_num +len(pub)
        else:
            test_author[key] = author_dict[key]
    # 保存成新文件
    dump_json(train_author, wfpath='../data/', wfname='train_set_author.json', indent=4)
    dump_json(test_author, wfpath='../data/', wfname='test_set_author.json', indent=4)




if __name__ == '__main__':
    # 数据预处理+划分数据集
    author_new,pub_new = pre_data(author_dict=load_json(rfdir='../data/', rfname='train_author.json'),
                                  pub_dict=load_json(rfdir='../data/', rfname='train_pub.json'))
                                #train_author_new.json和train_pub_new.json是处理后的数据集

    divide_data(author_new) # 划分数据集：train_set_author.json和test_set_author.json分别是训练和测试数据集


# class Singleton:
#     """
#     A non-thread-safe helper class to ease implementing singletons.
#     This should be used as a decorator -- not a metaclass -- to the
#     class that should be a singleton.
#
#     The decorated class can define one `__init__` function that
#     takes only the `self` argument. Also, the decorated class cannot be
#     inherited from. Other than that, there are no restrictions that apply
#     to the decorated class.
#
#     To get the singleton instance, use the `Instance` method. Trying
#     to use `__call__` will result in a `TypeError` being raised.
#
#     """
#
#     def __init__(self, decorated):
#         self._decorated = decorated
#
#     def Instance(self):
#         """
#         Returns the singleton instance. Upon its first call, it creates a
#         new instance of the decorated class and calls its `__init__` method.
#         On all subsequent calls, the already created instance is returned.
#
#         """
#         try:
#             return self._instance
#         except AttributeError:
#             self._instance = self._decorated()
#             return self._instance
#
#     def __call__(self):
#         raise TypeError('Singletons must be accessed through `Instance()`.')
#
#     def __instancecheck__(self, inst):
#         return isinstance(inst, self._decorated)
