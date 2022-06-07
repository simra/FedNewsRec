import argparse
from fl_training import GetUserDataFunc
from preprecoess import get_doc_input, get_test_input, get_train_input, load_matrix, parse_user, read_clickhistory, read_news
from model_pt import FedNewsRec
import torch
from torch import nn, optim
# from torchsummary import summary

root_data_path = '../../DP-REC/data' # MIND-Dataset Path
embedding_path = '../../DP-REC/wordvec' # Word Embedding Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
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

    model = FedNewsRec(title_word_embedding_matrix).cuda()
    # print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        for uinx in range(len(train_uid_table)):
            uid = train_uid_table[uinx]
            # print(uid)
            click, sample, label = get_user_data(uid)
            # print("@@@@", label)
            click = torch.from_numpy(click).cuda()
            sample = torch.from_numpy(sample).cuda()
            label = torch.from_numpy(label).type(torch.LongTensor).cuda()
            # print("####", label.type())

            optimizer.zero_grad()
            output = model(click, sample)
            # print(output.shape, label.shape)
            # print(output.cpu().detach().numpy(), label.cpu().detach().numpy())# output.item(), label.item())
            loss = criterion(output, torch.max(label, 1)[1])
            loss.backward()
            optimizer.step()

            print("Loss: ", loss.item())

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

