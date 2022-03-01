import pickle

with open('/mnt/nas6/users/wangweixuan/data/wmt14-ende/bin/dict.en.txt','r') as f:
    line = f.readlines()
vocab = []
i = 0
while i <len(line):
    vocab.append(line[i].split(" ")[0])
    i += 1

with open('/mnt/nas6/users/wangweixuan/ETM/scripts/stops.txt', 'r') as f:
    stops = f.read().split('\n')

vocab = [w for w in vocab if w not in stops]

with open('/mnt/nas6/users/wangweixuan/ETM/scripts/' + 'vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)