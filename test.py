import torch
from torch import nn
from torchtext.data.utils import get_tokenizer

from process import *
from model import *

PATH = "best_model.pt"

torch.load(PATH)
vocab = torch.load('vocab_obj.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = get_tokenizer('basic_english')

## Initiate an instance
ntokens = len(vocab)  # size of vocabulary
emsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 8  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

model.load_state_dict(torch.load(PATH))


# seqs = ["The man went to the store with his dog"]
# seqs = ["I have watched this [mask] and it was awesome"]
seq = "I have watched this movie and it was awesome"
seqs = [
               seq
        ]
# input_seq = [torch.tensor(vocab(tokenizer(seq)), dtype=torch.long)]
# print(input_seq)
# input_seq = torch.cat(tuple(filter(lambda t: t.numel() > 0, input_seq)))
# print(input_seq)
data = data_process(seqs, vocab, tokenizer)
# batch_data = batchify(data, 1, device)
# # print(batch_data)

print(torch.tensor(vocab(tokenizer(seq))))
print(seqs)
def predict(model, input_seq):
        model.eval()
        with torch.no_grad():
                src_mask = generate_square_subsequent_mask(len(input_seq)).to(device)
                out = model(input_seq.to(device), src_mask.to(device)) 
                out = out.view(-1, ntokens)
        return out
out = predict(model, data)
m = nn.Softmax(dim=1)
out = m(out)

# print(seqs)
# print(len(out))

predicts = out[2].topk(10)
# print(predicts[1])
for i in predicts[1]:
        print(vocab.lookup_token(i))