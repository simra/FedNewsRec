import numpy
import torch
import torch.nn as nn
#from torch.nn import MultiHeadAttention

from sklearn.metrics import accuracy_score, classification_report

npratio = 4


class AttentivePooling(nn.Module):
    def __init__(self, dim1: int, dim2: int):
        super(AttentivePooling, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

        self.dropout = nn.Dropout(0.2)
        self.dense  = nn.Linear(dim2, 200)
        self.tanh = nn.Tanh()
        self.flatten = nn.Linear(200, 1)
        self.softmax = nn.Softmax(dim=1)
       

    def forward(self, x):
        user_vecs = self.dropout(x)
        user_att = self.tanh(self.dense(user_vecs))
        #user_att = torch.squeeze(self.flatten(user_att))
        user_att = self.flatten(user_att)
        user_att = self.softmax(user_att)
        # todo: verify
        return torch.inner(user_vecs, user_att)

