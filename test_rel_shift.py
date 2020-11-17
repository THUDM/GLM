import torch
from mpu.transformer import GPT2ParallelSelfAttention

b = torch.arange(2) * 1000
h = torch.arange(3) * 100
pos_seq = torch.arange(10, -1, -1)
query = torch.arange(5) * 10
s = pos_seq.unsqueeze(0) + query.unsqueeze(1)
s = b.view(-1, 1, 1, 1) + h.view(1, -1, 1, 1) + s
s = GPT2ParallelSelfAttention._rel_shift_latest(s)
print(s)