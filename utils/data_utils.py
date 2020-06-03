import codecs
import json
from os.path import join
import pickle
import os
# from utils import settings


# def load_json(filename = '../data/valid/sna_valid_author_raw.json'):
def load_json(rfdir='../data/', rfname='train_pub.json'):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
    # with codecs.open(filename, 'r', encoding='utf-8') as rf:
        return json.load(rf)


# ''' "id":"","title":"","abstract":"","keywords": ["",""],
#     "authors": [{"name":"","org":""},{"name":"","org":""}],"venue": "","year": 2009
# '''
# pub_dict = ['id','title','abstract','keywords','authors','venue','year']
# def complement_pub(pub):# 补全论文数据（pub是load_json读出的字典）
#     n=0
#     for p in pub:# 每一篇论文
#         # print(n)
#         n=n+1
#         for item in pub_dict:
#             if item not in pub[p].keys():
#                 print(pub_dict[item])
# complement_pub(dic)

def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)

# def
pub_dic = load_json()
author_dic = load_json(rfname='train_author.json')



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


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)
