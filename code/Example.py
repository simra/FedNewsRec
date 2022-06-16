# %%
import os
import logging
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from utils import *
from preprecoess import *
from generator import *
from models import *
from fl_training import *
from datetime import datetime
from argparse import ArgumentParser

# %%
 
logging.basicConfig(level=logging.INFO)
root_data_path = '/home/rsim/MIND' # MIND-Dataset Path
embedding_path = '/home/rsim/GLOVE' # Word Embedding Path

config = {
    'lr': 0.3,
    'delta': 0.05,
    'lambd': 0,
    'num': 6,
    'optimizer': 'sgd'
    }
parser = ArgumentParser()
parser.add_argument('--lr', help='Learning rate', required=False, type=float)
parser.add_argument('--delta', help='Clipping parameter', required=False, type=float)
parser.add_argument('--num', help='Clients per round', required=False, type=int)
parser.add_argument('--optimizer', help='Optimizer type', required=False, choices=['sgd','adam'])

args=vars(parser.parse_args())
for k in config:
    if k in args and args[k] is not None:
        config[k] = args[k]


metrics_fn = f'metrics_{datetime.now().strftime("%y%m%d%H%M%S")}.tsv'
tfconfig = tf.ConfigProto()  
tfconfig.gpu_options.allow_growth=True  
session = tf.Session(config=tfconfig)
 
KTF.set_session(session)


# %%
# Read News
logging.info('Loading data')
news,news_index,category_dict,subcategory_dict,word_dict = read_news(root_data_path,['train','val'])
news_title,news_vert,news_subvert=get_doc_input(news,news_index,category_dict,subcategory_dict,word_dict)
title_word_embedding_matrix, have_word = load_matrix(embedding_path,word_dict)


# %%
#Parse User
logging.info('Preprocessing')
train_session, train_uid_click, train_uid_table = read_clickhistory(root_data_path,'train')
test_session, test_uid_click,test_uid_table = read_clickhistory(root_data_path,'val')
train_user = parse_user(train_session,news_index)
test_user = parse_user(test_session,news_index)
train_sess, train_user_id, train_label, train_user_id_sample = get_train_input(train_session,train_uid_click,news_index)
test_impressions, test_userids = get_test_input(test_session,news_index)

# %%


# %%
get_user_data = GetUserDataFunc(news_title,train_user_id_sample,train_user,train_sess,train_label,train_user_id)

# %%


# %%

timestamp = datetime.now().strftime("%y%m%d%H%M%S")
metrics_fn = f'metrics_{timestamp}.tsv'
config_fn = f'config_{timestamp}.json'

with open(config_fn, 'w', encoding='utf-8') as f:
    f.write(json.dumps(config)+'\n')


model, doc_encoder, user_encoder, news_encoder = \
    get_model(config['lr'], 
              config['delta'],
              title_word_embedding_matrix,
              optimizer_name=config['optimizer'])
Res = []
Loss = []
count = 0
while True:
    loss = fed_single_update(model,
                             doc_encoder,
                             user_encoder,
                             config['num'],
                             config['lambd'],
                             get_user_data,
                             train_uid_table)
    Loss.append(loss)
    if count % 25 == 0:
        news_scoring = news_encoder.predict(news_title,verbose=0)
        user_generator = get_hir_user_generator(news_scoring,test_user['click'],64)
        user_scoring = user_encoder.predict_generator(user_generator,verbose=0),
        user_scoring = user_scoring[0]
        g = evaluate(user_scoring,news_scoring,test_impressions)
        Res.append(g)
        #logging.info(f'{count*num}\t{list(g)}\t{loss}')
        #with open('FedRec-woLDP-1.json','a') as f:
        #    s = json.dumps(g) + '\n'
        #    f.write(s)
        metric_str = '\t'.join(map(str,g))
        out_str = f"{count*config['num']}\t{metric_str}\t{loss}"
        logging.info(out_str)
        with open(metrics_fn, 'a', encoding='utf-8') as f:
            f.write(out_str+"\n")
    count += 1

# %%


# %%



