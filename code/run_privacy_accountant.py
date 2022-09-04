from prv_accountant import Accountant

headers = ['gamma', 'clipping_norm', 'delta', 'noise_multiplier', 'clients_per_round', 'total_clients', 'auc', 'eps']
results = list(map(lambda a: dict(zip(headers, a)), 
                map(lambda a: map(float, a),
                    map(lambda a: map(lambda s: s.strip(), a),
                        map(lambda s: s.split('|'), """0 | 0.05 | 0.0001 | 0.2 | 1000 | 350000 | 0.561588 | 456.952 
0 | 0.05 | 0.001 | 0.2 | 1000 | 210000 | 0.560973 | 279.697 
0 | 0.01 | 0.0001 | 0.25 | 100 | 6000 | 0.557177 | 22.463 | 
0 | 0.01 | 0.0001 | 0.2 | 1000 | 200000 | 0.555957 | 276.905
0 | 0.01 | 0.001 | 0.25 | 1000 | 420000 | 0.554472 | 140.993
0 | 0.01 | 0.0001 | 0.25 | 1000 | 470000 | 0.552665 | 163.699
0 | 0.05 | 0.001 | 0.2 | 100 | 11000 | 0.552541 | 35.8017
0 | 0.01 | 0.0001 | 0.2 | 100 | 32000 | 0.551883 | 60.6105
0 | 0.05 | 0.001 | 0.25 | 1000 | 120000 | 0.549223 | 60.0202
0 | 0.05 | 0.001 | 0.25 | 100 | 2000 | 0.548946 | 15.1629
0 | 0.05 | 0.0001 | 0.25 | 100 | 3000 | 0.548297 | 20.4418
0 | 0.01 | 0.001 | 0.2 | 1000 | 410000 | 0.541351 | 519.761
0 | 0.05 | 0.0001 | 0.2 | 100 | 35000 | 0.536714 | 62.8389
0 | 0.01 | 0.001 | 0.2 | 100 | 12000 | 0.534319 | 36.5445
0 | 0.01 | 0.001 | 0.25 | 100 | 47000 | 0.534263 | 33.7333
0 | 0.05 | 0.0001 | 0.25 | 1000 | 80000 | 0.528747 | 58.4342""".split('\n'))))))

print(results)



#for clip_norm in [0.05, 0.01]:
#    for delta in [0.001, 0.0001]:
#        for noise_multiplier in [0.2, 0.25]:
#            for clients in [100, 1000]:
headers.append('eps_estimate')
for r in results:
    accountant = Accountant(
        noise_multiplier=r['noise_multiplier'],
        sampling_probability=r['clients_per_round']/50000,
        delta=r['delta'],
        eps_error=0.5,
        max_compositions=500
    )
    eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=int(r['total_clients']/r['clients_per_round']))
    r['eps_estimate'] = eps_estimate    
    print(' | '.join(map(str,map(lambda h: r[h], headers)))) #[r['clipping_norm'], r['delta'], r['noise_multiplier'], r['clients_per_round'], eps_low, eps_estimate, eps_upper])))