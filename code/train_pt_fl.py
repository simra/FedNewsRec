import argparse
from fl_training import GetUserDataFunc
from preprecoess import get_doc_input, get_test_input, get_train_input, load_matrix, parse_user, read_clickhistory, read_news
from model_pt import FedNewsRec
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
# from torchsummary import summary
from utils import evaluate, dcg_score, ndcg_score, mrr_score
from tqdm import tqdm 

#root_data_path = '../../DP-REC/data' # MIND-Dataset Path
#embedding_path = '../../DP-REC/wordvec' # Word Embedding Path
root_data_path = '/home/rsim/MIND' # MIND-Dataset Path
embedding_path = '/home/rsim/GLOVE' # Word Embedding Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--localiters', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--perround', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args()

    news,news_index,category_dict,subcategory_dict,word_dict = read_news(root_data_path,['train','val'])
    news_title,news_vert,news_subvert=get_doc_input(news,news_index,category_dict,subcategory_dict,word_dict)
    title_word_embedding_matrix, have_word = load_matrix(embedding_path,word_dict)
    train_session, train_uid_click, train_uid_table = read_clickhistory(root_data_path,'train')
    test_session, test_uid_click,test_uid_table = read_clickhistory(root_data_path,'val')
    train_user = parse_user(train_session,news_index)
    test_user = parse_user(test_session,news_index)
    train_sess, train_user_id, train_label, train_user_id_sample = get_train_input(train_session,train_uid_click,news_index)
    test_impressions, test_userids = get_test_input(test_session,news_index)
    get_user_data = GetUserDataFunc(news_title,train_user_id_sample,train_user,train_sess,train_label,train_user_id)

    # print(news_title.shape)
    # news_title = torch.from_numpy(news_title).cuda()

    model = FedNewsRec(title_word_embedding_matrix).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # print(torch.cuda.memory_summary())

    # doc cache
    doc_cache = []
    print('building doc_cache')
    for j in range(len(news_title)):
        doc_cache.append(torch.from_numpy(np.array([news_title[j]])))

    for ridx in range(args.rounds):
        random_index = np.random.permutation(len(train_uid_table))[:args.perround]
        pretrained_dict = model.state_dict()
        running_average = model.state_dict()
        total_loss = 0.

        for uidx in random_index:
            uid = train_uid_table[uidx]
            click, sample, label = get_user_data(uid)
            click = torch.from_numpy(click).cuda()
            print(click.shape)
            sample = torch.from_numpy(sample).cuda()
            label = torch.from_numpy(label).type(torch.LongTensor).cuda()

            for itr in range(args.localiters):
                output, _, _ = model(click, sample, label)
                # print(output.shape, label.shape)
                # print(output.cpu().detach().numpy(), label.cpu().detach().numpy())# output.item(), label.item())
                loss = criterion(output, torch.max(label, 1)[1])
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                del output
                torch.cuda.empty_cache()

            # TODO: keep track of the differences
            update = {layer: (model.state_dict()[layer] - pretrained_dict[layer]) for layer in pretrained_dict}
            running_average = {layer: running_average[layer] + update[layer] / args.perround for layer in update}
            # TODO: reset model weights
            model.load_state_dict(pretrained_dict)

            del click, sample, label
            torch.cuda.empty_cache()

        model.load_state_dict(running_average)

        del pretrained_dict, running_average
        torch.cuda.empty_cache()

        # print(torch.cuda.memory_summary())

        print("Loss:", total_loss / args.localiters / args.perround)

        if ridx % 25 == 0:
            AUC = []
            MRR = []
            nDCG5 = []
            nDCG10 =[]
            for i in tqdm(range(len(test_impressions))):
                #print(i)
                docids = test_impressions[i]['docs']
                labels = test_impressions[i]['labels']
                nv_imp = []
                for j in docids:
                    nv_imp.append(doc_cache[j])
                nv = model.news_encoder(torch.stack(nv_imp).squeeze(1).cuda()).detach().cpu().numpy()                    
                #nv = np.array(nv_imp)
                nv_hist = []                
                for j in test_user['click'][i]:
                    nv_hist.append(doc_cache[j])
                    # print(j)
                nv_hist = model.news_encoder(torch.stack(nv_hist).squeeze(1).cuda())
                # print("nv_hist:", nv_hist.shape)
                uv = model.user_encoder(nv_hist.unsqueeze(0)).detach().cpu().numpy()[0]

                score = np.dot(nv,uv)
                auc = roc_auc_score(labels,score)
                mrr = mrr_score(labels,score)
                ndcg5 = ndcg_score(labels,score,k=5)
                ndcg10 = ndcg_score(labels,score,k=10)

                AUC.append(auc)
                MRR.append(mrr)
                nDCG5.append(ndcg5)
                nDCG10.append(ndcg10)
            print(np.mean(AUC), np.mean(MRR), np.mean(nDCG5), np.mean(nDCG10))
"""
            news_encodings = []
            for title in news_title:
                title = torch.from_numpy(np.array([title])).cuda()
                news_encodings.append(model.news_encoder(title).cpu())
            user_encodings = model.user_encoder(news_encodings[test_user['click'][:args.batchsize]])
            result = evaluate(user_encodings, news_encodings, test_impressions)
            print("Test Log:", result)
"""
