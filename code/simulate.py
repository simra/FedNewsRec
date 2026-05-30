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
# from prv_accountant import Accountant
import privacy
from functools import reduce
from copy import deepcopy

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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lmb)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.gamma)

    print('Using GPU:', torch.cuda.is_available(), torch.cuda.current_device())
    os.makedirs(args.output_path, exist_ok=True)
    if args.metrics_format == 'date':
        metrics_fn = os.path.join(args.output_path, f'metrics_{datetime.now().strftime("%y%m%d%H%M%S")}.tsv')
    else:
        metrics_fn = os.path.join(args.output_path, f'metrics_{args.lr}_{args.gamma}_{args.lmb}_{args.delta}_{args.noise_multiplier}_{args.clip_norm}_{args.perround}_{args.rounds}.tsv')
    with open(metrics_fn, 'w', encoding='utf-8') as f:
        f.write(json.dumps(vars(args))+'\n')

    # doc cache
    doc_cache = []
    # print('building doc_cache')
    for j in range(len(news_title)):
        doc_cache.append(torch.from_numpy(np.array([news_title[j]])))

    metrics_keys = ['total_clients', 'auc', 'eps', 'mrr', 'ndcg@5', 'ndcg@10']
    metrics = dict(zip(metrics_keys, [0]*len(metrics_keys)))
    
    # verify we have a valid configuration. Return empty metrics if we don't
    if (args.noise_multiplier > 0. or args.clip_norm > 0. or args.delta > 0.) and (args.clip_norm <= 0. or args.noise_multiplier <= 0. or args.delta <= 0.):
        print("Error: invalid DP configuration. Bailing.")
        return metrics


    # TODO: we need to change 50000 to an input argument
    """
    accountant = None
    if args.noise_multiplier > 0. and args.clip_norm > 0.:
        try:
            accountant = Accountant(
                    noise_multiplier=args.noise_multiplier,
                    sampling_probability=args.perround/50000,
                    delta=args.delta,
                    eps_error=0.1,
                    max_compositions=args.rounds*args.perround  # TODO: verify this is the right argument. Or should it just be args.rounds?
            )
        except:
            print('Warning: failed to create accountant')
    """

    for ridx in range(args.rounds):
        random_index = np.random.permutation(len(train_uid_table))[:args.perround]
        pretrained_dict = {name: p.clone().detach() for name, p in model.named_parameters() } # deepcopy(model.state_dict())
        running_average = None
        b_sum = 0. # adaptive clipping count of clients less than the norm
        nu_c = 1.  # adaptive clipping learning rate
        sigma_b = 0.01  # DP noise to add to estimated fraction of clients not clipped

        total_loss = 0.
        total_weight = 0.

        model.train()
        requires_dp_noise = { layer: p.requires_grad for layer, p in model.named_parameters()}                
        for uidx in random_index:
            uid = train_uid_table[uidx]
            click, sample, label = get_user_data(uid)
            click = torch.from_numpy(click).cuda(args.device)
            sample = torch.from_numpy(sample).cuda(args.device)
            label = torch.from_numpy(label).cuda(args.device)
            client_loss = 0

            for itr in range(args.localiters):
                output, _ = model(click, sample)
                loss = criterion(output, label)
                total_loss += loss.item()
                client_loss += loss.item()
                if total_loss / args.localiters / args.perround > args.divergence_threshold:
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
            
            if args.agg_method == 'dga':
                weight = 1.0/(1.0+np.exp(client_loss))
                update = {layer: update[layer] * weight for layer in update }    
                total_weight += weight
            else:
                total_weight += 1

            #sparseness = {layer: (requires_dp_noise.get(layer, "Unknown"), float(torch.count_nonzero(update[layer]).cpu().item())/torch.numel(update[layer])) for layer in update}
            #print(json.dumps(sparseness, indent=3))
            if args.clip_norm > 0.0:
                update_norm = torch.sqrt(reduce(lambda a, b: a + torch.square(torch.norm(b, p=2)), update.values(), 0.))
                # print("Update norm:", update_norm)
                if update_norm > args.clip_norm:
                    scale = args.clip_norm / update_norm
                    update = {layer: scale * update[layer] for layer in update}
                else:
                    b_sum += 1
            if args.quantize_scale != -1.:
                update = {layer: torch.round(args.quantize_scale * update[layer]) for layer in update}
            # TODO: add noise
            if args.noise_multiplier != -1.:
                # raise NotImplementedError('Add skellam mechanism here')
                # Instead we will do global DP
                pass
            # TODO: modular clipping
            if args.bitwidth != -1:
                # TODO: we need to translate this to the positive half axis
                update = {layer: (torch.remainder(update[layer] + 2**(args.bandwidth-1), 2**args.bitwidth) - 2**(args.bandwidth-1)) / args.quantize_scale for layer in update} 
            if running_average is None:
                running_average = {layer: update[layer] / args.perround for layer in update}
            else:
                running_average = {layer: running_average[layer] + update[layer] / args.perround for layer in update}
            model.load_state_dict(pretrained_dict, strict=False)

            del click, sample, label
            torch.cuda.empty_cache()

        # We do local clipping for each client, but we'll add noise globally
        """
        if args.clip_norm != float('inf'):
            update_norm = torch.sqrt(reduce(lambda a, b: a + torch.square(torch.norm(b, p=2)), running_average.values(), 0.))
            print("Update norm:", update_norm)
            if update_norm > args.clip_norm:
                scale = args.clip_norm / update_norm
                running_average = {layer: scale * running_average[layer] for layer in running_average}
        """
        if args.noise_multiplier > 0 and args.clip_norm > 0:
            running_average = {layer: (running_average[layer] 
                                       + torch.normal(mean=torch.zeros_like(running_average[layer]), std=args.noise_multiplier*args.clip_norm*torch.ones_like(running_average[layer]))
                                       ) if requires_dp_noise.get(layer, True)
                                       else running_average[layer] for layer in running_average}
        
        updated_dict = {layer: pretrained_dict[layer] + running_average[layer] for layer in running_average}
        model.load_state_dict(updated_dict, strict=False)
        
        del pretrained_dict, running_average
        torch.cuda.empty_cache()
        # update the learning rate
        scheduler.step()

        print("Round:", ridx+1, "Loss:", total_loss / args.localiters / args.perround)
        eps_estimate = np.inf
        if args.noise_multiplier > 0. and args.clip_norm > 0.:            
            eps_estimate = privacy.update_privacy_accountant(args.noise_multiplier, args.clip_norm, args.delta,
                                                             num_clients=len(train_uid_table), curr_iter=ridx, 
                                                             num_clients_curr_iter=args.perround)
        
        if args.adaptive_clip_gamma >= 0.:
            b_sum = (b_sum + np.random.normal(0.0, sigma_b))/args.perround
            # old_clip_norm = args.clip_norm
            args.clip_norm = args.clip_norm * np.exp(-nu_c*(b_sum-args.adaptive_clip_gamma))
            # print(old_clip_norm, args.clip_norm, b_sum, b_sum-args.adaptive_clip_gamma)
        
            """
            if accountant is None:
                # the accountant was unsuccessfully generated
                eps_estimate = np.nan
            else:
                try:                
                    eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=ridx+1)
                    print("Epsilon:", eps_estimate)
                except:
                    print('Warning: accountant failed')
                    eps_estimate = np.nan
            """
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
                total_clients = (ridx+1)*args.perround
                metrics_out = [total_clients, np.mean(AUC), eps_estimate, np.mean(MRR), np.mean(nDCG5), np.mean(nDCG10)]                
                metric_str = '\t'.join(map(str,metrics_out))
                out_str = f"{total_clients}\t{metric_str}\t{total_loss / args.localiters / args.perround}"
                print(out_str)
                with open(metrics_fn, 'a', encoding='utf-8') as f:
                    f.write(out_str+"\n")
                metrics_out = dict(zip(metrics_keys,metrics_out))
                if metrics_out['auc']>metrics['auc']:
                    metrics = metrics_out
                    print('New best: {}'.format(metrics))
    return metrics

def ray_helper(args):
    # convert ray dict to namespace
    args = argparse.Namespace(**args)
    metrics = main(args)
    ray.tune.report(**metrics)

class TrialTerminationReporter(CLIReporter):
    def __init__(self, max_progress_rows=100):
        super(TrialTerminationReporter, self).__init__(max_progress_rows=max_progress_rows)
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
    parser.add_argument('--agg_method', default='fedavg', choice=['fedavg', 'dga'])
    parser.add_argument('--noise_multiplier', type=float, default=0., help='local noise multiplier')
    parser.add_argument('--clip_norm', type=float, default=-1.0, help='L2 clip norm for distributed DP mechanism')
    parser.add_argument('--delta', type=float, default=-1.0, help='delta in (eps, delta)-DP')
    parser.add_argument('--quantize_scale', type=float, default=-1., help='quantize_scale in secure aggregation')
    parser.add_argument('--bitwidth', type=int, default=-1, help='bitwidth to transmit the local updates')
    parser.add_argument('--adaptive_clip_gamma', type=float, default=-1, help='target quantile for adaptive clipping (-1 to disable)')
    parser.add_argument('--divergence_threshold', type=float, default=100, help='loss threshold for when to bail on training')
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
