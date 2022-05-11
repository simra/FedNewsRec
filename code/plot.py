import matplotlib.pyplot as plt
import json
import sys

metrics = []
# 'FedRec-woLDP-1.json'
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    for l in f:
        #metrics.append(json.loads(l))
        metrics.append(list(map(float,l.strip().split('\t'))))

header = ['Clients', 'AUC', 'MRR', 'NDCG@5', 'NDCG@10', 'loss']

fig, ax = plt.subplots()

iters = [m[0] for m in metrics]
for i,h in enumerate(header[1:-1]):
    ax.plot(iters, [m[i+1] for m in metrics], label=h)

ax.legend()
plt.savefig('plot.png')


