import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BiLSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(BiLSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        # self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        x = self.embeddings(sentence)
        lstm_out, (self.lstm_h, self.lstm_c) = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        print("input: sentence = ", sentence)
        print("x = self.embeddings(sentence): ", self.embeddings(sentence), "self.embeddings(sentence).size(): ",
              self.embeddings(sentence).size())
        print("lstm_out, (self.lstm_h, self.lstm_c) = self.lstm(x, self.hidden)")
        print("lstm_out", lstm_out, 'lstm_out.size()', lstm_out.size())
        print("lstm_out", lstm_out[-1], 'lstm_out[-1].size()', lstm_out[-1].size())
        print("lstm_h", self.lstm_h, "h.size(): ", self.lstm_h.size())
        print("lstm_c", self.lstm_c, "c.size(): ", self.lstm_c.size())
        print("y = self.hidden2label(lstm_out[-1]): ", y, " y.size(): ", y.size())
        print("return > log_probs = F.log_softmax(y): ", F.log_softmax(y),"log_probs .size(): ", F.log_softmax(y).size())
        return log_probs
