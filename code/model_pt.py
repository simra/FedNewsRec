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

class SwapTrailingAxes(nn.Module):
    def __init__(self):
        super(SwapTrailingAxes, self).__init__()
        
    def forward(self, x):        
        return x.mT

class DocEncoder(nn.Module):
    def __init__(self):        
        super(DocEncoder,self).__init__()
        self.phase1 = nn.Sequential(
            nn.Dropout(0.2),
            SwapTrailingAxes(),            
            nn.Conv1d(300,400,3),
            nn.ReLU(),
            nn.Dropout(0.2),
            SwapTrailingAxes()
        )
        self.attention = nn.MultiheadAttention(400,20)
        self.phase2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            AttentivePooling(30,400)
        )
    
    def forward(self, x):
        # print('phase1: ', x.dtype)
        l_cnnt = self.phase1(x)
        # print('doc_encoder:phase1',l_cnnt.shape)
        l_cnnt, attention_weights = self.attention(l_cnnt, l_cnnt, l_cnnt)
        # print('doc_encoder:attention', l_cnnt.shape)
        result = self.phase2(l_cnnt)
        # print('doc_encoder:phase2', result.shape)
        return result

    def get_weights(self):
        return [p.detach().clone() for p in self.parameters()]
    
    def set_weights(self, weights, do_grad=False):
        for p,w in zip(self.parameters(), weights):
            p.data = w.data
            if do_grad:
                p.grad = w.grad


class VecTail(nn.Module):
    def __init__(self, n):
        super(VecTail, self).__init__()
        self.n = n

    def forward(self, x):
        return x[:, -self.n:, :]

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
        self.gru2 = nn.GRU(400,400, batch_first=True)
        self.pool3 = AttentivePooling(2, 400)

    def forward(self, news_vecs_input):    
        #news_vecs =self.dropout1(news_vecs_input)
        #gru_input = self.tail(news_vecs)
        #vec1 = self.gru(gru_input)
        #vecs2 = self.attention(*[news_vecs]*3)
        #vec2 = self.pool(vecs2)
        #print('news_vecs_input', news_vecs_input.shape)
        user_vecs2, _u_weights = self.attention2(*[news_vecs_input]*3)
        user_vecs2 = self.dropout2(user_vecs2)
        user_vec2 = self.pool2(user_vecs2)
        #print('pool2_user_vec2', user_vec2.shape)
        #user_vec2 = keras.layers.Reshape((1,400))(user_vec2)
        #user_vec2 = user_vec2.unsqueeze(1)

        user_vecs1 = self.tail2(news_vecs_input)
        #print('tail2_user_vecs1', user_vecs1.shape)
        user_vec1, _u_hidden = self.gru2(user_vecs1)
        #print('gru2_user_vec1', user_vec1.shape)
        user_vec1 = user_vec1[:, -1, :]
        #user_vec1 = keras.layers.Reshape((1,400))(user_vec1)
        #user_vec1 = user_vec1.unsqueeze(1)
        
        user_vecs = torch.stack([user_vec1, user_vec2], dim=1) #keras.layers.Concatenate(axis=-2)([user_vec1,user_vec2])
        print(user_vecs.shape)
        vec = self.pool3(user_vecs)
        print(vec.shape)
        return vec

    def get_weights(self):
        return [p.detach().clone() for p in self.parameters()]
    
    def set_weights(self, weights, do_grad=False):
        for p,w in zip(self.parameters(), weights):
            p.data = w.data
            if do_grad:
                p.grad = w.grad


class TimeDistributed(nn.Module):    
    def __init__(self, module): #, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        # self.batch_first = batch_first

    def forward(self, x):
        #print('TimeDist_x',x.size())
        if len(x.size()) <= 2:
            return self.module(x)

        output = torch.tensor([])
        for i in range(x.size(1)):
          output_t = self.module(x[:, i, :, :])
          output_t  = output_t.unsqueeze(1)
          output = torch.cat((output, output_t ), 1)
        #print('TimeDist_output', output.size())
        return output
        # # Squash samples and timesteps into a single axis
        # x_reshape = x.contiguous().view(x.size(0), -1, x.size(-1))  # (samples * timesteps, input_size)
        #print('TimeDist_x_reshape',x_reshape.shape)
        # y = self.module(x_reshape)
        # print('TimeDist_y', y.shape)
        # # We have to reshape Y
        # if self.batch_first:
        #     y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        # else:
        #    y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        # print('TimeDist_y_reshape',y.size())
        #return y

class FedNewsRec(nn.Module):
    def __init__(self, title_word_embedding_matrix):
        super(FedNewsRec, self).__init__()
        self.doc_encoder = DocEncoder() 
        self.user_encoder = UserEncoder()
        # TODO: should this embedding matrix be frozen?
        self.title_word_embedding_layer = nn.Embedding.from_pretrained(torch.tensor(title_word_embedding_matrix), freeze=False)
    
        # click_title = Input(shape=(50,30),dtype='int32')
        # can_title = Input(shape=(1+npratio,30),dtype='int32')
    
        self.softmax = nn.Softmax(dim=1)
        self.click_td = TimeDistributed(self.doc_encoder) #, batch_first=True)
        self.can_td = TimeDistributed(self.doc_encoder) #, batch_first=True)
        
    def forward(self, click_title, can_title):
        
        #print('can_title: ', can_title.shape)
        click_word_vecs = self.title_word_embedding_layer(click_title)
        #print('click',click_word_vecs.shape, click_word_vecs.dtype)
        can_word_vecs = self.title_word_embedding_layer(can_title)
        #print('can', can_word_vecs.shape, can_word_vecs.dtype)
        click_vecs = self.click_td(click_word_vecs)
        #print('click_vecs (None, 50, 400)', click_vecs.shape)
        can_vecs = self.can_td(can_word_vecs)
        #print('can_vecs (None, 5, 400)', can_vecs.shape)
    
        user_vec = self.user_encoder(click_vecs)        
        #print('user_vec (None, 400)', user_vec.shape)
        # TODO verify
        scores = torch.einsum('ijk,ik->ij',  can_vecs, user_vec)
        #print('scores  (None, 5)', scores.shape)
        logits = self.softmax(scores)     
        #print('logits  (None, 5)', logits.shape)
                
        
        #print('user_vec', user_vec.shape)
        #print('news_vec', news_vec.shape)
        return logits, user_vec

    def news_encoder(self, news_input):
        news_word_vecs = self.title_word_embedding_layer(news_input)
        news_vec = self.doc_encoder(news_word_vecs)
        return news_vec