import torch
from torch import nn
from torchtext.data.utils import get_tokenizer


from model import *
PATH = "best_model.pt"
torch.load(PATH)
vocab = torch.load('vocab_obj.pth')

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
        
        return out
print(seq)
out = predict(model, input_seq)
out = out.view(-1, ntokens)
m = nn.LogSoftmax(dim=1)
print(m(out[0]))
