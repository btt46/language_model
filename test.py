import math
from numpy import size
import torch
from torch import ThroughputBenchmark, nn, Tensor
import torch.nn.functional as F 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from model import *
PATH = "best_model.pt"
torch.load(PATH)
vocab = torch.load('vocab_obj.pth')

print(vocab.get_itos()[:10])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



tokenizer = get_tokenizer('basic_english')



## Initiate an instance
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

model.load_state_dict(torch.load(PATH))


seq = "The man went to the store with [MASK] dog"
input_seq = torch.tensor(vocab(tokenizer(seq)), dtype=torch.long)
print(input_seq)

def predict(model, input_seq):
        model.eval()
        src_mask = generate_square_subsequent_mask(len(input_seq)).to(device)
        out = model(input_seq.to(device), src_mask.to(device))
        output = out.view(-1, ntokens)
        return output

out = predict(model, input_seq)
print(seq)
for i in range(len(out)):
        predict_val = out[i]
        probs = torch.softmax(predict_val , dim=-1) 
        word_idx = torch.multinomial(probs, num_samples=1).item()
        if i < 10:
                print(word_idx, vocab.lookup_token(word_idx))

