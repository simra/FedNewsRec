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
        user_att = self.flatten(user_att)
        user_att = self.softmax(user_att)
        result = torch.einsum('ijk,ijk->ik', user_vecs, user_att)
        return result

class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.permute(*self.dims)

class DocEncoder(nn.Module):
    def __init__(self):
        #sentence_input = Input(shape=(30,300), dtype='float32')
        super(DocEncoder,self).__init__()
        self.phase1 = nn.Sequential(
            nn.Dropout(0.2),
            Permute(0,2,1),            
            nn.Conv1d(300,400,3),
            nn.ReLU(),
            nn.Dropout(0.2),
            Permute(0,2,1)
        )
        self.attention = nn.MultiheadAttention(400,20)
        self.phase2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            AttentivePooling(30,400)
        )
    
    def forward(self, x):
        l_cnnt = self.phase1(x)
        print(l_cnnt.shape)
        l_cnnt, attention_weights = self.attention(l_cnnt, l_cnnt, l_cnnt)
        print(l_cnnt.shape)
        result = self.phase2(l_cnnt)
        print(result.shape)
        return result


class UserEncoder(nn.Module):
    def __init__(self):        
        super(UserEncoder,self).__init__()
        # news_vecs_input = Input(shape=(50,400), dtype='float32')
        #self.dropout1 = nn.Dropout(0.2)
        #self.tail = VecTail(15)
        #self.gru = nn.GRU(400, 400)
        #self.attention = nn.MultiheadAttention(400, 20)
        #self.pool = AttentivePooling(50, 400)
        self.attention2 = nn.MultiheadAttention(400, 20)
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = AttentivePooling(50, 400)
        self.tail2 = VecTail(20)
        self.gru2 = nn.GRU(400,400)
        self.pool3 = AttentivePooling(2, 400)

    def forward(self, news_vecs_input):    
        #news_vecs =self.dropout1(news_vecs_input)
        #gru_input = self.tail(news_vecs)
        #vec1 = self.gru(gru_input)
        #vecs2 = self.attention(*[news_vecs]*3)
        #vec2 = self.pool(vecs2)
    
        user_vecs2, _u_weights = self.attention2(*[news_vecs_input]*3)
        user_vecs2 = self.dropout2(user_vecs2)
        user_vec2 = self.pool2(user_vecs2)
        print(user_vec2.shape)
        #user_vec2 = keras.layers.Reshape((1,400))(user_vec2)
        #user_vec2 = user_vec2.unsqueeze(1)

        user_vecs1 = self.tail2(news_vecs_input)
        user_vec1, _u_hidden = self.gru2(user_vecs1)
        #user_vec1 = keras.layers.Reshape((1,400))(user_vec1)
        #user_vec1 = user_vec1.unsqueeze(1)
        
        user_vecs = torch.stack([user_vec1, user_vec2], dim=1) #keras.layers.Concatenate(axis=-2)([user_vec1,user_vec2])
        print(user_vecs.shape)
        vec = self.pool3(user_vecs)
        print(vec.shape)
        return vec
