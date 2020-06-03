import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np



class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))#初始化
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
    def forward(self,input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


if __name__ == '__main__':
    wf_name = 'aminer'
    emb_model = EmbeddingModel.Instance()
    emb_model.train(wf_name)
    print('loaded')