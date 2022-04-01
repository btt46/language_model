import math
import torch
from torch import nn, Tensor
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text = ("After Abraham Lincoln won the November 1860 presidential "
        "election on an anti-slavery platform, an initial seven "
        "slave states declared their secession from the country "
        "to form the Confederacy. War broke out in April 1861 "
        "when secessionist forces attacked Fort Sumter in South "
        "Carolina, just over a month after Lincoln's "
        "inauguration.") 

tokenizer = get_tokenizer('basic_english')

print(vocab(tokenizer("After Abraham Lincoln won the [MASK] 1860 presidential ")))

## Initiate an instance
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

model.load_state_dict(torch.load(PATH))
model.eval()

src_mask = generate_square_subsequent_mask(56).to(device)

result = model(torch.tensor(vocab(tokenizer(text))).to(device),src_mask)
output_flat = result.view(-1, ntokens)
print(len(output_flat), len(output_flat[0]))
