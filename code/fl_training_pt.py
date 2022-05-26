import numpy as np
import torch

def GetUserDataFunc(news_title,train_user_id_sample,train_user,train_sess,train_label,train_user_id):
    def _get_user_data(uid):
        click = []
        sample = []
        label = []
        for sid in train_user_id_sample[uid]:
            click.append(train_user['click'][train_user_id[sid]])
            sample.append(train_sess[sid])
            label.append(train_label[sid])
        click = np.array(click)
        sample = np.array(sample)
        label = np.array(label)
        click = news_title[click]
        sample = news_title[sample]        
        return click,sample,label
    return _get_user_data


def add_noise(weights,lambd):
    for i in range(len(weights)):
        weights[i] += np.random.laplace(scale = lambd,size=weights[i].shape)
    return weights

def loss_fn(y_pred, y_true):
    return (-y_pred.log() * y_true).sum(dim=1).mean()


def fed_single_update(model,doc_encoder,user_encoder,num,lambd,get_user_data,train_uid_table):
    random_index = np.random.permutation(len(train_uid_table))[:num]
    
    all_news_weights = []
    all_user_weights = []
    old_news_weight = doc_encoder.get_weights()
    old_user_weight = user_encoder.get_weights()
    print('old news weights:',[o.shape for o in old_news_weight])
    print('old user weights:', [u.shape for u in old_user_weight])
    sample_nums = []
    
    loss_history = []

    for uinx in random_index:
        doc_encoder.set_weights(old_news_weight)
        user_encoder.set_weights(old_user_weight)

        uid = train_uid_table[uinx]
        click, sample, label = get_user_data(uid)
        print('inputs: ', click.dtype, sample.dtype)
        print('label: ', label)
        click = torch.from_numpy(click)
        sample = torch.from_numpy(sample)
        label = torch.from_numpy(label)
        
        model.optimizer.zero_grad()

        # Make predictions for this batch
        logits, user_vec = model(click, sample)
        print(logits.shape, label.shape)
        loss = loss_fn(logits, label)
        loss.backward()

        # Adjust learning weights
        model.optimizer.step()
        
        loss_history.append(loss.detach().numpy())
        news_weight = doc_encoder.get_weights()
        user_weight = user_encoder.get_weights()
        if lambd>0:
            news_weight = add_noise(news_weight, lambd)
            user_weight = add_noise(user_weight, lambd)
        #noise = 
        #weight += noise
        all_news_weights.append(news_weight)
        all_user_weights.append(user_weight)
        sample_nums.append(label.shape[0])
    
    sample_nums = np.array(sample_nums)
    sample_nums = sample_nums/sample_nums.sum()
    
    # doc_weights = [np.average(weights.numpy(), axis=0,weights=sample_nums) for weights in zip(*all_news_weights)]
    # user_weights = [np.average(weights.numpy(), axis=0,weights=sample_nums) for weights in zip(*all_user_weights)]
    # TODO: make this a GPU operation
    doc_weights = map(torch.from_numpy, [np.average([w.numpy() for w in weights], axis=0, weights=sample_nums) for weights in zip(*all_news_weights)])
    user_weights = map(torch.from_numpy, [np.average([w.numpy() for w in weights], axis=0, weights=sample_nums) for weights in zip(*all_user_weights)])
    #print('weights_numpy:', doc_weights.shape, user_weights.shape)
    doc_encoder.set_weights(doc_weights)
    user_encoder.set_weights(user_weights)
    # TODO: also a gpu op
    loss = np.array(loss_history).mean()
    print('average loss',loss)
    return loss
