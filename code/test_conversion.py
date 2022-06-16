import torch
import torch.nn as nn

import tensorflow as tf
import keras
from keras.layers import *
from keras.models import Model
from model_pt import Attention as ptAttention, DocEncoder as ptDocEncoder, TimeDistributed as ptTimeDistributed, UserEncoder as ptUserEncoder




class ptAttentivePoolingTest(nn.Module):
    def __init__(self, dim1, dim2, dim3, keras_dense):
        super(ptAttentivePoolingTest, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
        self.dense = nn.Linear(dim2, dim3)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(dim3, 1)
        self.softmax = nn.Softmax(dim=1)

        keras_weights = keras_dense.layers[1].get_weights()
        # print(keras_weights)
        self.dense.weight.data = torch.tensor(keras_weights[0]).transpose(0,1)
        self.dense.bias.data = torch.tensor(keras_weights[1])

        keras_weights = keras_dense.layers[2].get_weights()
        # print(keras_weights)
        self.dense2.weight.data = torch.tensor(keras_weights[0]).transpose(0,1)
        self.dense2.bias.data = torch.tensor(keras_weights[1])

    @staticmethod
    def tfAttentivePooling(dim1, dim2, dim3):
        vecs_input = Input(shape=(dim1, dim2), dtype='float32')
        user_att = Dense(dim3, activation='tanh', kernel_initializer='random_uniform', bias_initializer='random_uniform')(vecs_input)
        user_att = keras.layers.Flatten()(Dense(1, kernel_initializer= 'random_uniform', bias_initializer= 'random_uniform')(user_att))
        user_att = Activation('softmax')(user_att) 
        user_att = keras.layers.Dot((1,1))([vecs_input,user_att])
        model = Model(vecs_input, user_att)
        return model   
    
    def forward(self, x):
        result = self.softmax(self.dense2(self.tanh(self.dense(x))).squeeze(2))
        print(x.shape, result.shape)
        result = torch.einsum('ijk,ij->ik', x, result)
        return result

    @staticmethod
    def test(dim1, dim2, dim3):
        tfd = ptAttentivePoolingTest.tfAttentivePooling(dim1, dim2, dim3)
        for layer in tfd.layers:
            print(layer.name, layer.output_shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ptd = ptAttentivePoolingTest(dim1, dim2, dim3, tfd)
            
            x = torch.randn((2, dim1, dim2))

            x_tf = tf.convert_to_tensor(x)

            keras = tfd(x_tf).eval()
            with torch.no_grad():
                ptd.eval()
                pt = ptd(x).detach().numpy()
                print(keras.shape, pt.shape)
                print('error:', np.linalg.norm(keras-pt))



class ptAttn(nn.Module):
    def __init__(self, input_shape, embed_dim, heads, tfAttention):
        super(ptAttn, self).__init__()
        self.input_shape = input_shape  # should be x by embed_dim        
        
        self.attention = ptAttention(input_shape[1], heads, embed_dim//heads)
        self.attention.fromTensorFlow(tfAttention)
        

    def forward(self,x):
        print(x.shape, self.input_shape)
        return self.attention([x]*3)

    
    @staticmethod
    def tfAttentionTest(input_shape, heads, num_per_head):
        from models import Attention
        vecs_input = Input(shape=input_shape)
        vecs_attn = Attention(heads, num_per_head)([vecs_input, vecs_input, vecs_input])
        return Model(vecs_input, vecs_attn)

    @staticmethod
    def ptAttentionTest(input_shape, embed_dim, heads):
        tfd = ptAttn.tfAttentionTest(input_shape, heads, embed_dim//heads)
        for layer in tfd.layers:
            print(layer.name, layer.output_shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ptd = ptAttn(input_shape, embed_dim, heads, tfd)
            
            x = torch.randn((5, input_shape[0], input_shape[1]))

            x_tf = tf.convert_to_tensor(x)

            keras = tfd(x_tf).eval()
            with torch.no_grad():
                ptd.eval()
                ptresult = ptd(x)
                pt = ptresult.detach().numpy()
                print(keras.shape, pt.shape)
                print('error:', np.linalg.norm(keras-pt))



class ptDocument(nn.Module):
    def __init__(self, tfDoc):
        super(ptDocument, self).__init__()
        
        self.docencoder = ptDocEncoder()
        self.docencoder.fromTensorFlow(tfDoc)
        

    def forward(self,x):
        return self.docencoder(x)

    
    @staticmethod
    def tfDocEncoder():
        from models import Attention as tfAttention
        sentence_input = Input(shape=(30,300), dtype='float32')
        
        l_cnnt = Conv1D(400,3,activation='relu')(sentence_input)
    
        l_cnnt = tfAttention(20,20)([l_cnnt,l_cnnt,l_cnnt])
        l_cnnt = keras.layers.Activation('relu')(l_cnnt)
    
        title_vec = ptAttentivePoolingTest.tfAttentivePooling(30, 400, 200)(l_cnnt)
        sentEncoder = Model(sentence_input, title_vec)
        return sentEncoder

    @staticmethod
    def ptDocEncoderTest():
        tfd = ptDocument.tfDocEncoder()
        for layer in tfd.layers:
            print(layer.name, layer.output_shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ptd = ptDocument(tfd)
            
            x = torch.randn((5, 30, 300))

            x_tf = tf.convert_to_tensor(x)

            keras = tfd(x_tf).eval()
            with torch.no_grad():
                ptd.eval()
                ptresult = ptd(x)
                pt = ptresult.detach().numpy()
                print(keras.shape, pt.shape)
                print('error:', np.linalg.norm(keras-pt))



class ptTimeDist(nn.Module):
    def __init__(self, tfTD, doc_encoder):
        super(ptTimeDist, self).__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.docencoder = ptDocEncoder().to(device)
        for l in tfTD.layers:
            print(l.name, l.output_shape)
        self.docencoder.fromTensorFlow(doc_encoder)
        self.tdTD = ptTimeDistributed(self.docencoder)
        

    def forward(self,x):
        return self.tdTD(x)

    
    @staticmethod
    def tfTimeDistributed():
        doc_encoder = ptDocument.tfDocEncoder()
        embedding_vecs = Input(shape=(2,30,300))
        click_vecs = TimeDistributed(doc_encoder)(embedding_vecs)
        return Model(embedding_vecs, click_vecs), doc_encoder



    @staticmethod
    def ptTimeDistributedTest():
        x = torch.randn((1, 50, 30, 300)).cuda()
        
        
        tfd, doc_encoder = ptTimeDist.tfTimeDistributed()
        for layer in tfd.layers:
            print(layer.name, layer.output_shape)
            if 'time' in layer.name.lower():
                print(layer.layer.name)
                weights=layer.get_weights()
                print([w.shape for w in weights])
                
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ptd = ptTimeDist(tfd, doc_encoder)
            
            x_tf = tf.convert_to_tensor(x.detach().cpu())

            keras = tfd(x_tf).eval()
            print(type(keras))
        with torch.no_grad():
            ptd.eval()
            x.cuda()
            ptresult = ptd(x.cuda())
            pt = ptresult.detach().cpu().numpy()
            print(keras.shape, pt.shape)
            print('error:', np.linalg.norm(keras-pt))




class ptUserEnc(nn.Module):
    def __init__(self, tfU):
        super(ptUserEnc, self).__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #self.docencoder = ptDocEncoder().to(device)
        #for l in tfTD.layers:
        #    print(l.name, l.output_shape)
        #self.docencoder.fromTensorFlow(doc_encoder)
        self.tdUE = ptUserEncoder().to(device) #(self.docencoder)
        self.tdUE.fromTensorFlow(tfU)

    def forward(self,x):
        return self.tdUE(x)

    
    @staticmethod
    def tfUserEncoder():
        news_vecs_input = Input(shape=(50,400), dtype='float32')
        from models import Attention as tfAttention
        user_vecs2 = tfAttention(20, 20)([news_vecs_input]*3)        
        user_vec2 = ptAttentivePoolingTest.tfAttentivePooling(50, 400, 400)(user_vecs2)
        user_vec2 = keras.layers.Reshape((1,400))(user_vec2)
            
        user_vecs1 = Lambda(lambda x:x[:,-20:,:])(news_vecs_input)
        user_vec1 = GRU(400, recurrent_activation='sigmoid', kernel_initializer="ones", recurrent_initializer="ones", bias_initializer="ones")(user_vecs1)
        user_vec1 = keras.layers.Reshape((1,400))(user_vec1)

        user_vecs = keras.layers.Concatenate(axis=-2)([user_vec1,user_vec2])
        print(user_vecs.shape)
        vec = ptAttentivePoolingTest.tfAttentivePooling(2, 400, 400)(user_vecs)
            
        sentEncodert = Model(news_vecs_input, vec)
        return sentEncodert


    @staticmethod
    def ptUserEncoderTest():
        x = torch.randn((1, 50, 400)).cuda()
    
        tfU = ptUserEnc.tfUserEncoder()
        for layer in tfU.layers:
            print(layer.name, layer.output_shape)
        #    if 'time' in layer.name.lower():
        #        print(layer.layer.name)
        #        weights=layer.get_weights()
        #        print([w.shape for w in weights])
                
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ptU = ptUserEnc(tfU)
            
            x_tf = tf.convert_to_tensor(x.detach().cpu())

            keras = tfU(x_tf).eval()
            print(keras.shape)
        with torch.no_grad():
            ptU.eval()
            x.cuda()
            ptresult = ptU(x.cuda())
            pt = ptresult.detach().cpu().numpy()
            print(keras.shape, pt.shape)
            print('error:', np.linalg.norm(keras-pt))



class ptGRU(nn.Module):
    def __init__(self, tfG):
        super(ptGRU, self).__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #self.docencoder = ptDocEncoder().to(device)
        #for l in tfTD.layers:
        #    print(l.name, l.output_shape)
        #self.docencoder.fromTensorFlow(doc_encoder)
        self.gru = nn.GRU(input_size=400, hidden_size=400, num_layers=1, batch_first=True, bidirectional=False).to(device) #(self.docencoder)
        names = [weight.name for layer in tfG.layers for weight in layer.weights]
        
        weights = tfG.layers[1].get_weights()
        print([(n,w.shape) for (n,w) in zip(names, weights)])
        for p in self.gru.named_parameters():
            s1 = p[1].data.shape
            if p[0] == 'weight_ih_l0':                        
                p[1].data = torch.tensor(weights[0]).transpose(0,1).contiguous().cuda()
            elif p[0] == 'weight_hh_l0':
                p[1].data = torch.tensor(weights[1]).transpose(0,1).contiguous().cuda()
            elif p[0] == 'bias_ih_l0':
                p[1].data = torch.tensor(weights[2]).cuda()
            elif p[0] == 'bias_hh_l0':
                p[1].data = torch.zeros(p[1].data.shape).cuda()
            print(p[0], s1, p[1].shape)

    def forward(self,x):
        g,_ = self.gru(x)
        return g #[:, -1, :]

    
    @staticmethod
    def tfGRU():
        news_vecs_input = Input(shape=(15,400), dtype='float32')
        #user_vec1 = GRU(400,  return_sequences=True)(news_vecs_input)
        user_vec1 = GRU(400, return_sequences=True, activation='tanh', recurrent_activation='sigmoid')(news_vecs_input) #, kernel_initializer="ones", recurrent_initializer="ones", bias_initializer="ones")(news_vecs_input)
        sentEncodert = Model(news_vecs_input, user_vec1)
        return sentEncodert


    @staticmethod
    def ptGRUTest():
        x = torch.randn((1, 15, 400)).cuda()
    
        tfU = ptGRU.tfGRU()
        for layer in tfU.layers:
            print(layer.name, layer.output_shape)
        #    if 'time' in layer.name.lower():
        #        print(layer.layer.name)
        #        weights=layer.get_weights()
        #        print([w.shape for w in weights])
                
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ptU = ptGRU(tfU)
            
            x_tf = tf.convert_to_tensor(x.detach().cpu())

            keras = tfU(x_tf).eval()
            print(keras.shape)
        with torch.no_grad():
            ptU.eval()
            x.cuda()
            ptresult = ptU(x.cuda())
            pt = ptresult.detach().cpu().numpy()
            print(keras.shape, ptresult.shape)
            print('error:', [np.linalg.norm(keras[:,i,:]-pt[:,i,:]) for i in range(ptresult.shape[1])])



#ptAttentivePoolingTest.test(2, 3, 4)
#ptAttn.ptAttentionTest((3,20),400,20)
#ptDocument.ptDocEncoderTest()
#ptTimeDist.ptTimeDistributedTest()
ptUserEnc.ptUserEncoderTest()
#ptGRU.ptGRUTest()