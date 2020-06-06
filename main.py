import torch
import torch.nn as nn
import torch.nn.functional as F
import triplet_dataset
import bert
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 10
WEIGHT_DECAY = 0.1
LR = 1e-3


def train():
    model = bert.TextNet(code_length=32)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    dataset = triplet_dataset.TripletDataset(dtype='train')
    data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_loss = 0
    train_acc = 0

    loss = nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for idx, item in enumerate(data):
        # print(len(item[0]))
        print(idx)
        optimizer.zero_grad()

        (normal, positive, negative) = tuple(item)


        item_embedding = bert.get_embedding(list(normal), textNet=model)#.cuda()
        positive_embedding = bert.get_embedding(list(positive), textNet=model)#.cuda()
        negative_embedding = bert.get_embedding(list(negative), textNet=model)#.cuda()

        output = loss(item_embedding, positive_embedding, negative_embedding)
        output.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(idx, output)


if __name__ == '__main__':
    train()
