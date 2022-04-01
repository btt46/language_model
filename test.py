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

print(vocab)

text = ("After Abraham Lincoln won the November 1860 presidential "
        "election on an anti-slavery platform, an initial seven "
        "slave states declared their secession from the country "
        "to form the Confederacy. War broke out in April 1861 "
        "when secessionist forces attacked Fort Sumter in South "
        "Carolina, just over a month after Lincoln's "
        "inauguration.") 

tokenizer = get_tokenizer('basic_english')

model = TransformerModel()
model.load_state_dict()
model.eval()

