import torch
import torch.nn as nn

import tensorflow as tf
import keras
from keras.layers import *
from keras.models import Model
from model_pt import Attention as ptAttention, DocEncoder as ptDocEncoder




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


ptAttentivePoolingTest.test(2, 3, 4)
ptAttn.ptAttentionTest((3,20),400,20)
ptDocument.ptDocEncoderTest()

