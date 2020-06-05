import torch
import torch.nn as nn
import torch.nn.functional as F
import triplet_dataset
import bert
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 1
WEIGHT_DECAY = 0.1
LR = 1e-3

def train():

    model = bert.TextNet(code_length=32)
    dataset = triplet_dataset.TripletDataset(dtype='train')
    data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_loss = 0
    train_acc = 0

    loss = nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for idx, item in enumerate(data):
        # print(len(item[0]))
        print(idx)
        (normal, positive, negative) = tuple(item)
        # print(list(normal))
        if len(normal[0]) > 2000 or len(positive[0]) > 2000 or len(negative[0]) > 2000:
            continue
        print(len(normal[0]))
        print(len(positive[0]))
        print(len(negative[0]))
        item_embedding = bert.get_embedding(list(normal), textNet=model)
        positive_embedding = bert.get_embedding(list(positive), textNet=model)
        negative_embedding = bert.get_embedding(list(negative), textNet=model)



        if idx > 32:
            break



if __name__ == '__main__':
    train()