import torch
from torch.utils.data import DataLoader, Dataset
import json
import random


class TripletDataset(Dataset):

    def __init__(self, dtype='train'):
        if dtype == 'train':
            with open('./data/train_set_author.json') as f:
                self.author_dict = json.loads(f.read())
        else:
            with open('./data/test_set_author.json') as f:
                self.author_dict = json.loads(f.read())

        # 将json文件转换为list格式
        # {author：
        #           id：
        #               article}
        # ->
        # [author, id, article]
        self.author = []

        for author, author_dict in self.author_dict.items():
            for author_id, author_id_list in author_dict.items():
                for article in author_id_list:
                    self.author.append([author, author_id, article])

        with open('./data/train_pub_new.json') as f:
            self.pub = json.loads(f.read())

    def __len__(self):

        return len(self.author)

    def __getitem__(self, idx):
        # author, author_id, article
        item = self.author[idx]
        (author, author_id, article) = tuple(item)
        sample_success = True

        # 随机抽取当前item的正例和负例
        while sample_success:

            negative_author_id = random.sample(self.author_dict[author].keys(), 1)[0]
            negative_article = random.sample(self.author_dict[author][negative_author_id], 1)[0]
            positive_article = random.sample(self.author_dict[author][author_id], 1)[0]

            if negative_author_id != author_id and article != positive_article:
                # print(self.article2sentence(article))
                # print([article, positive_article, negative_article])
                return [self.article2sentence(article),
                        self.article2sentence(positive_article),
                        self.article2sentence(negative_article)]
                # return [article, positive_article, negative_article]

    def article2sentence(self, article_id):
        article = self.pub[article_id]
        return article['title'] + ' ' + article['abstract'] + ' ' + ' '.join(article['keywords'])


# data = TripletDataset()
# print(data[0])
