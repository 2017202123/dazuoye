from os.path import join
# import keras.backend as K
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Bidirectional
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim

LMDB_NAME = "author_100.emb.weighted"
lc = LMDBClient(LMDB_NAME)

data_cache = {}


# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
#
#
# def root_mean_log_squared_error(y_true, y_pred):
#     first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
#     second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
#     return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


# def create_model():
#     model = Sequential()
#     model.add(Bidirectional(LSTM(64), input_shape=(300, 100)))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#
#     model.compile(loss="msle",
#                   optimizer='rmsprop',
#                   metrics=[root_mean_squared_error, "accuracy", "msle", root_mean_log_squared_error])
#
#     return model

# def root_mean_squared_error(y_true, y_pred):
#     return torch.sqrt(torch.mean((y_pred - y_true)**2, axis=-1))
#     # return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
#
# def root_mean_log_squared_error(y_true, y_pred):
#     first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
#     second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
#     return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def MLSE(y_true, y_pred):
    y_true_log = torch.log(1. + y_true)
    y_pred_log = torch.log(1. + y_pred)
    return torch.mean(nn.MSELoss(y_pred_log,y_true_log), axis=-1)

class RNN_model(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim):
        super(RNN_model,self).__init__()
        # self.doc_embedding_lookup = doc_embedding
        self.doc_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        #########################################
        # 其中输入特征大小是 "doc_embedding_dim"
        #    输出特征大小是 "lstm_hidden_dim"
        # 这里的LSTM应该有两层，并且输入和输出的tensor都是(batch, seq, feature)大小
        # batch_first=True,输入为(batch, seq, feature)
        self.rnn_lstm = nn.LSTM(input_size = self.doc_embedding_dim,
                                hidden_size = self.lstm_dim,
                                num_layers = 2,
                                batch_first=True)
        ##########################################
        self.fc = nn.Linear(self.lstm_dim, 1) # 输出为一个数字K
        nn.init.xavier_uniform_(self.fc.weight)


    def forward(self,batch_doc,batch_size,is_test = False):
        # batch_input = self.doc_embedding_lookup(sentence).view(batch_size,-1,self.doc_embedding_dim)
        ################################################
        # 这里你需要将上面的"batch_input"输入到你在rnn模型中定义的lstm层中
        # lstm的隐藏层输出应该被定义叫做变量"output", 初始的隐藏层(initial hidden state)和记忆层(initial cell state)应该是0向量.
        output, (hn, cn) = self.rnn_lstm(batch_doc,None) # None 表示 hidden state 会用全0的 state
        ################################################
        out = output.contiguous().view(-1,self.lstm_dim)
        out = self.fc(out)   #out.size: (batch_size * sequence_length ,vocab_length)
        # if is_test:
        #     #测试阶段(或者说生成诗句阶段)使用
        #     prediction = out[ -1, : ].view(1,-1)
        #     output = prediction
        # else:
        #     #训练阶段使用
        #    output = out
        output = out
        return output

# train_author 里面k最大为464,
def sampler(clusters, k=300, batch_size=10, min=1, max=300, flatten=False):
    xs, ys = [], []
    for b in range(batch_size):
        num_clusters = np.random.randint(min, max)
        sampled_clusters = np.random.choice(len(clusters), num_clusters, replace=False)
        items = []
        for c in sampled_clusters:
            items.extend(clusters[c])
        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)] # 每一个cluster选k个样本，有重复
        x = []
        for p in sampled_points:
            x.append(p)
            # if p in data_cache:
            #     x.append(data_cache[p])
            # else:
            #     print("a")
            #     x.append(lc.get(p))
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))
        ys.append(num_clusters)
    return np.stack(xs), np.stack(ys)

def run_training(mink = 1,maxk = 500,WORD_EMBEDDING_DIM=10, LSTM_HIDDEN_DIM=64,epochs=1000,steps=100,batch_size=1000):
    # 处理数据集
    # poems_vector, word_to_int, int_to_word = process_poems('./subset_poems.txt')
    # 建立word embedding层(用于根据word index得到对应word的embedding表示，或者说向量表示)
    # word_embedding = rnn.word_embedding( vocab_length= len(word_to_int) , embedding_dim= WORD_EMBEDDING_DIM)
    # 建立RNN模型,之前建立的word embedding层作为它的一部分
    rnn_model = RNN_model(    embedding_dim= WORD_EMBEDDING_DIM,
                              lstm_hidden_dim=LSTM_HIDDEN_DIM)
    optimizer = optim.Adam(rnn_model.parameters(), lr=3e-3)
    # Criterion = MLSE()
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  #如果你想读取之前训练得到的模型并接着进行训练，请去掉这一行前面的#号

    '''读取文件并处理！！！！！！！！！！！！！！！！！'''
    clusters = []
    for epoch in range(epochs):
        # 对每个batch进行训练
        # for batch_num in range(n_chunk):
        for step in range(steps):
            batch_x,batch_y = sampler(clusters, k=300, batch_size=batch_size, min=mink, max=maxk, flatten=False)
            batch_x = torch.LongTensor(batch_x) #batch_x size: (batch_size, sequence_length)
            '''???????? batch_y的形状是什么'''
            batch_y = torch.LongTensor(batch_y).view(-1) #batch_y size: (batch_size * sequence_length)
            # batch_size = batch_x.size()[0]
            pre = rnn_model(batch_x, batch_size = batch_size)
            # loss = Criterion(pre,batch_y)
            loss = MLSE(batch_y, pre)
            print("epoch  ",epoch,'step',step,"loss is: ", loss.tolist())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)  #梯度截断(按照模)
            optimizer.step()
            optimizer.zero_grad()
        torch.save(rnn_model.state_dict(), './poem_generator_rnn')
        print("finish  save model of epoch : {}!".format(epoch))
        # print("epoch using time {:.3f}".format(time.time()-start_time))


def gen_K(test_file,WORD_EMBEDDING_DIM=10, LSTM_HIDDEN_DIM=64,):
    # poems_vector, word_int_map, int_word_map = process_poems('./subset_poems.txt')
    # word_embedding = rnn.word_embedding(vocab_length=len(word_int_map) , embedding_dim=WORD_EMBEDDING_DIM)
    rnn_model = RNN_model(embedding_dim=WORD_EMBEDDING_DIM,
                              lstm_hidden_dim=LSTM_HIDDEN_DIM)
    rnn_model.load_state_dict(torch.load('./poem_generator_rnn')) #读取训练得到的模型

    # 指定开始的字
    poem = start_token + begin_word
    word = begin_word
    while word != end_token:
        # input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        # input = torch.from_numpy(input)
        output = rnn_model(input, batch_size = 1, is_test=True)
        # word = to_word(output.data.numpy()[-1], int_word_map, poem)
        # poem += word
        # if len(poem) > 50:
        #     break
    return poem


def gen_train(clusters, k=300, batch_size=1000, flatten=False):
    while True:
        yield sampler(clusters, k, batch_size, flatten=flatten)


def gen_test(k=300, flatten=False):
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    xs, ys = [], []
    names = []
    for name in name_to_pubs_test:
        names.append(name)
        num_clusters = len(name_to_pubs_test[name])
        x = []
        items = []
        for c in name_to_pubs_test[name]:  # one person
            for item in name_to_pubs_test[name][c]:
                items.append(item)
        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
        for p in sampled_points:
            if p in data_cache:
                x.append(data_cache[p])
            else:
                x.append(lc.get(p))
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))
        ys.append(num_clusters)
    xs = np.stack(xs)
    ys = np.stack(ys)
    return names, xs, ys


def run_rnn(k=300, seed=1106):
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')
    test_names, test_x, test_y = gen_test(k)
    np.random.seed(seed)
    clusters = []
    for domain in name_to_pubs_train.values():
        for cluster in domain.values():
            clusters.append(cluster)
    for i, c in enumerate(clusters):
        if i % 100 == 0:
            print(i, len(c), len(clusters))
        for pid in c:
            data_cache[pid] = lc.get(pid)
    model = create_model()
    # print(model.summary())
    model.fit_generator(gen_train(clusters, k=300, batch_size=1000), steps_per_epoch=100, epochs=1000,
                        validation_data=(test_x, test_y))
    kk = model.predict(test_x)
    wf = open(join(settings.OUT_DIR, 'n_clusters_rnn.txt'), 'w')
    for i, name in enumerate(test_names):
        wf.write('{}\t{}\t{}\n'.format(name, test_y[i], kk[i][0]))
    wf.close()






if __name__ == '__main__':
    run_rnn()
