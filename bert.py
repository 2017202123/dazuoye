from pytorch_transformers import BertModel, BertConfig, BertTokenizer
import torch.nn as nn
import torch


# ——————构造模型——————
class TextNet(nn.Module):
    def __init__(self, code_length):  # code_length为fc映射到的维度大小
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('./data/bert-base-uncased-config.json')
        self.textExtractor = BertModel.from_pretrained(
            './data/bert-base-uncased-pytorch_model.bin', config=modelConfig)
        # self.textExtractor.eval()
        embedding_dim = self.textExtractor.config.hidden_size

        # self.fc = nn.Linear(embedding_dim, code_length)
        # self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        # features = self.fc(text_embeddings)
        # features = self.tanh(features)
        return text_embeddings


def get_embedding(texts, textNet):
    # textNet = TextNet(code_length=32)

    # ——————输入处理——————
    tokenizer = BertTokenizer.from_pretrained('./data/bert-base-uncased-vocab.txt')

    # texts = ["Henson is a pig",
    # "[CLS] Jim [SEP]"]
    tokens, segments, input_masks = [], [], []
    for text in texts:
        text = '[CLS]' + text + '[SEP]'
        tokenized_text = tokenizer.tokenize(text)  # 用tokenizer对句子分词
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    max_len = max([len(single) for single in tokens])  # 最大的句子长度

    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding
    # segments列表全0，因为只有一个句子1，没有句子2
    # input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
    # 相当于告诉BertModel不要利用后面0的部分

    # 转换成PyTorch tensors
    tokens_tensor = torch.tensor(tokens)
    segments_tensors = torch.tensor(segments)
    input_masks_tensors = torch.tensor(input_masks)

    # ——————提取文本特征——————
    text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)  # text_hashCodes是一个32-dim文本特征
    # print(text_hashCodes)
    return text_hashCodes


# textnet = TextNet(code_length=32)
# get_embedding(texts=[''])