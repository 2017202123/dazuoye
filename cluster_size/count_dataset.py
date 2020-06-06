import torch
from torch.utils.data import DataLoader, Dataset
import json
import random
from utils.data_utils import load_json
import numpy as np


class CountDataset(Dataset):

    def __init__(self,dtype = 'train',maxk=500,):
        self.dtye = dtype
        self.seq_len = maxk
        self.maxk = maxk

        self.clusters = []

        # 将pub_emb转换成clusters

        # for author, author_dict in self.author_dict.items():
        #     for author_id, author_id_list in author_dict.items():
        #         for article in author_id_list:
        #             self.author.append([author, author_id, article])
        #
        # with open('./data/train_pub_new.json') as f:
        #     self.pub = json.loads(f.read())
        if dtype == 'train':
            pub_emb = load_json(rfdir='../data/', rfname='pub_emb.json')
            authors = load_json(rfdir='../data/', rfname='train_set_author.json')

        for author in authors:
            for nameid  in authors[author]:
                doc_set =[]
                for pid in authors[author][nameid]:
                    doc_set.append(pub_emb[pid])
                self.clusters.append(doc_set)

    def __len__(self):

        # return len(self.clusters)
        return self.maxk

    def __getitem__(self,index):

        num_clusters = index+1 #np.random.randint(self.mink, self.maxk)
        sampled_clusters = np.random.choice(len(self.clusters), num_clusters, replace=False)
        items = []
        for c in sampled_clusters:
            items.extend(self.clusters[c])
        sampled_points = [items[p] for p in np.random.choice(len(items), self.seq_len, replace=True)]  # 从所有的cluster中选择z个样本，有重复
        x = []
        for p in sampled_points:
            x.append(p)
        # x = torch.LongTensor(np.stack(x))
        # y = torch.LongTensor(num_clusters)
        x = np.stack(x)
        y = np.stack([num_clusters])
        # return [torch.LongTensor(np.stack(x)),torch.LongTensor(num_clusters)]
        return [x,y]



# data = TripletDataset()
# print(data[0])
