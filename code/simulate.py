import argparse
from preprocess import GetUserDataFunc, get_doc_input, get_test_input, get_train_input, load_matrix, parse_user, read_clickhistory, read_news
from model import FedNewsRec
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from utils import evaluate, dcg_score, ndcg_score, mrr_score
from tqdm import tqdm 
from datetime import datetime
import sys
import os
import json
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.trial import Trial
from prv_accountant import Accountant
from functools import reduce
from copy import deepcopy
from accountant import skellam_epsilon

#@ray.remote(num_gpus=1)
def main(args):
    print(vars(args))
    root_data_path = args.data_path
    embedding_path = args.embedding_path

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

    model = FedNewsRec(title_word_embedding_matrix).cuda(args.device)
    model_size = reduce(lambda a, b: a + np.prod(list(b.size())), model.state_dict().values(), 0)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lmb)
    criterion = nn.CrossEntropyLoss()

    print('Using GPU:', torch.cuda.is_available(), torch.cuda.current_device())
    os.makedirs(args.output_path, exist_ok=True)
    if args.metrics_format == 'date':
        metrics_fn = os.path.join(args.output_path, f'metrics_{datetime.now().strftime("%y%m%d%H%M%S")}.tsv')
    else:
        metrics_fn = os.path.join(args.output_path, f'metrics_{args.lr}_{args.gamma}_{args.lmb}_{args.perround}_{args.rounds}.tsv')
    with open(metrics_fn, 'w', encoding='utf-8') as f:
        f.write(json.dumps(vars(args))+'\n')

    # doc cache
    doc_cache = []
    # print('building doc_cache')
    for j in range(len(news_title)):
        doc_cache.append(torch.from_numpy(np.array([news_title[j]])))

    metrics_keys = ['auc', 'mrr', 'ndcg@5', 'ndcg@10']
    metrics = dict(zip(metrics_keys, [0,0,0,0]))
    for ridx in range(args.rounds):
        random_index = np.random.permutation(len(train_uid_table))[:args.perround]
        pretrained_dict = deepcopy(model.state_dict())
        running_average = None
        total_loss = 0.

        model.train()        
        for uidx in random_index:
            uid = train_uid_table[uidx]
            click, sample, label = get_user_data(uid)
            click = torch.from_numpy(click).cuda(args.device)
            sample = torch.from_numpy(sample).cuda(args.device)
            label = torch.from_numpy(label).cuda(args.device)

            for itr in range(args.localiters):
                output, _ = model(click, sample)
                loss = criterion(output, label)
                total_loss += loss.item()
                if total_loss / args.localiters / args.perround > 1e6:
                    model.eval()               
                    with torch.no_grad():
                        print('model output:', output.detach().cpu().numpy(), label.detach().cpu().numpy())
                        for _ in range(3):
                            print('other model output:', model(click, sample)[0].detach().cpu().numpy())                                        
                    return metrics
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                del output
                torch.cuda.empty_cache()

            update = {layer: model.state_dict()[layer] - pretrained_dict[layer] for layer in pretrained_dict}
            # print("Update before skellam:", update['doc_encoder.phase1.2.weight'])
            if args.noise_multiplier > 0.:
                assert args.clip_l2_norm != float('inf'), 'As skellam mechanism is enabled, `clip_l2_norm` is required.'
                assert args.quantize_scale != -1., 'As skellam mechanism is enabled, `quantize_scale` is required.'
                assert args.bitwidth != -1, 'As skellam mechanism is enabled, `bitwidth` is required.'
                update_l2_norm = torch.sqrt(reduce(lambda a, b: a + torch.square(torch.norm(b, p=2)), update.values(), 0.))
                # print("Update norm:", update_norm)
                if update_l2_norm > args.clip_l2_norm:
                    update = {layer: args.clip_l2_norm / update_l2_norm * update[layer] for layer in update}
                # print("Update after clipping:", update['doc_encoder.phase1.2.weight'])
                update = {layer: torch.round(args.quantize_scale * update[layer]).long() for layer in update}
                # print("Update after quantization:", update['doc_encoder.phase1.2.weight'])
                poisson_rate = 2. * (args.noise_multiplier * args.clip_l2_norm)**2
                update = {layer: (update[layer] + torch.poisson(poisson_rate * torch.ones_like(update[layer])) - torch.poisson(poisson_rate * torch.ones_like(update[layer]))).long() for layer in update}
                # print("Update after noise:", update['doc_encoder.phase1.2.weight'])
                update = {layer: (torch.remainder(update[layer] + 2**(args.bitwidth-1), 2**args.bitwidth) - 2**(args.bitwidth-1)) / args.quantize_scale for layer in update} 
                # print("Update after skellam", update['doc_encoder.phase1.2.weight'])
            if running_average is None:
                running_average = {layer: update[layer] / args.perround for layer in update}
            else:
                running_average = {layer: running_average[layer] + update[layer] / args.perround for layer in update}
            model.load_state_dict(pretrained_dict)

            del click, sample, label
            torch.cuda.empty_cache()

        updated_dict = {layer: pretrained_dict[layer] + running_average[layer] for layer in running_average}
        model.load_state_dict(updated_dict)
        
        del pretrained_dict, running_average
        torch.cuda.empty_cache()

        print("Round:", ridx+1, "Loss:", total_loss / args.localiters / args.perround)
        if args.noise_multiplier > 0.:
            eps = skellam_epsilon(scale = args.quantize_scale, 
                            central_stddev = 2 * np.sqrt(args.perround) * args.noise_multiplier * args.clip_l2_norm,
                            l2_sens = 2 * args.clip_l2_norm,
                            beta = 0.,
                            dim = model_size,
                            q = args.perround / 50000,
                            steps = ridx+1,
                            delta = args.delta
            )
            print("Epsilon:", eps)
        sys.stdout.flush()
        
        if (ridx + 1) % args.checkpoint == 0:
            print('running test metrics')
            model.eval()
            with torch.no_grad():
                AUC = []
                MRR = []
                nDCG5 = []
                nDCG10 =[]
                for i in range(len(test_impressions)):
                    if i%10000==0:
                        print('.', end='') 
                        sys.stdout.flush()
                    docids = test_impressions[i]['docs']
                    labels = test_impressions[i]['labels']
                    nv_imp = [doc_cache[j] for j in docids]
                    nv = model.news_encoder(torch.stack(nv_imp).squeeze(1).cuda(args.device)).detach().cpu().numpy()                    
                    nv_hist = [doc_cache[j] for j in test_user['click'][i]]            
                    nv_hist = model.news_encoder(torch.stack(nv_hist).squeeze(1).cuda(args.device))
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
                print()
                metrics_out = [np.mean(AUC), np.mean(MRR), np.mean(nDCG5), np.mean(nDCG10)]                
                metric_str = '\t'.join(map(str,metrics_out))
                if args.noise_multiplier > 0.:
                    eps = skellam_epsilon(scale = args.quantize_scale,
                                    central_stddev = 2 * np.sqrt(args.perround) * args.noise_multiplier * args.clip_l2_norm,
                                    l2_sens = 2 * args.clip_l2_norm,
                            	    beta = 0.,
                            	    dim = model_size,
                            	    q = args.perround / 50000,
                            	    steps = ridx+1,
                            	    delta = args.delta
                    )
                    out_str = f"{(ridx+1)*args.perround}\t{eps}\t{metric_str}\t{total_loss / args.localiters / args.perround}"
                else:
                    out_str = f"{(ridx+1)*args.perround}\t{metric_str}\t{total_loss / args.localiters / args.perround}"
                print(out_str)
                with open(metrics_fn, 'a', encoding='utf-8') as f:
                    f.write(out_str+"\n")
                if metrics_out[0]>metrics['auc']:
                    metrics = dict(zip(metrics_keys,metrics_out))
    return metrics

def ray_helper(args):
    # convert ray dict to namespace
    args = argparse.Namespace(**args)
    metrics = main(args)
    ray.tune.report(**metrics)

class TrialTerminationReporter(CLIReporter):
    def __init__(self, max_progress_rows=100):
        super(TrialTerminationReporter, self).__init__(max_progress_rows)
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated or done

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--localiters', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lmb', type=float, default=0)
    parser.add_argument('--checkpoint', type=int, default=25)
    #parser.add_argument('--device', type=int, default=None, required=False)
    parser.add_argument('--perround', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--data_path', default='/mnt/fednewsrec/data') 
    parser.add_argument('--embedding_path', default='/mnt/fednewsrec/wordvec')     
    parser.add_argument('--output_path', default='.')
    parser.add_argument('--sweep', required=False, help='path to ray sweep config')
    parser.add_argument('--metrics_format', default='date', choices=['date', 'config'], help='whether for format the metrics filename by date or configuration')
    parser.add_argument('--noise_multiplier', type=float, default=0., help='local noise multiplier, main power switch for DP.')
    parser.add_argument('--clip_l2_norm', type=float, default=float('inf'), help='L2 clip norm for distributed DP mechanism')
    parser.add_argument('--delta', type=float, default=1e-3, help='delta in (eps, delta)-DP')
    parser.add_argument('--quantize_scale', type=float, default=-1., help='quantize_scale in secure aggregation')
    parser.add_argument('--bitwidth', type=int, default=-1, help='bitwidth to transmit the local updates')
    args = parser.parse_args()

    # assert (quantize_scale == -1.) != (bitwidth == -1), 'quantize_scale and bitwidth must both be -1 or not -1 simultaneously to guarantee correctness.'

    #if args.device is None and torch.cuda.is_available():
    #    args.device=torch.cuda.current_device() 
    device = -1
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    setattr(args,'device', device)

    torch.cuda.set_device(args.device)
    
    if args.sweep is None:
        main(args)
    else:
        ray.init()
        # TODO: data loading can happen once, globally. In most cases we will run one job per process, but it might be useful to support multiple jobs per process.       
        with open(args.sweep, 'r', encoding='utf-8') as inF:
            sweep = json.load(inF)

        sweep_config=vars(args)

        for k in sweep:
            sweep_config[k] = tune.grid_search(sweep[k])

        batch_job = tune.run(
            ray_helper,
            config=sweep_config,
            resources_per_trial={'gpu': 1},
            progress_reporter=TrialTerminationReporter()
        )
