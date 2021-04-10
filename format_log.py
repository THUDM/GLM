import sys
import json

data = []

with open(sys.argv[1]) as f:
    hyper = ''
    for line in f:
        if line[0] == '{':
            res = json.loads(line)
            data.append((hyper, res))
        else:
            hyper = line.strip()

keys = [key for key in data[0][1] if type(data[0][1][key]) is not str]
print(keys)
for hyper, res in data:
    print(hyper, end='\t')
    print('\t'.join([f'{res[key]:.4f}' for key in keys]))

