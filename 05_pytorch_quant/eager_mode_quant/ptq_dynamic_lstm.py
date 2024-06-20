import time
import urllib3

import torch.nn as nn
import torch.quantization
from io import open
from utils import *


"""
官方示例：(beta) Dynamic Quantization on an LSTM Word Language Model
地址：https://github.com/pytorch/tutorials/blob/main/advanced_source/dynamic_quantization_tutorial.py
"""


######################################################################
# 1. 定义模型
# -------------------
class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


######################################################################
# 2. 加载数据集
# Wikitext-2 in assets/data/Wikitext-2
# ------------------------
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


model_data_filepath = '../../assets/data/'

corpus = Corpus(model_data_filepath + 'wikitext-2')

######################################################################
# 3. 加载预训练模型
# 下载预训练文件(109M)：https://s3.amazonaws.com/pytorch-tutorial-assets/word_language_model_quantize.pth
# 并放入assets/data
# -----------------------------
ntokens = len(corpus.dictionary)

model = LSTMModel(
    ntoken=ntokens,
    ninp=512,
    nhid=256,
    nlayers=5,
)

pretrained_pth = model_data_filepath + 'word_language_model_quantize.pth'


def download_pth(path):
    """ Download a pretrained model from s3 and put it in assets/data"""
    if not os.path.exists(path):
        # 创建连接池
        print(f'{path} is not exists \nDownloading pretrained')
        http = urllib3.PoolManager()
        url = 'https://s3.amazonaws.com/pytorch-tutorial-assets/word_language_model_quantize.pth'
        # 发送 GET 请求下载文件
        with open(path, 'wb') as f:
            with http.request('GET', url, preload_content=False) as resp:
                for chunk in resp.stream():
                    f.write(chunk)


if not os.path.exists(pretrained_pth):
    download_pth(pretrained_pth)

model.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu')
    )
)

model.eval()
print(f'原始网络结构： {model}')

######################################################################
# 测试模型
input_ = torch.randint(ntokens, (1, 1), dtype=torch.long)
hidden = model.init_hidden(1)
temperature = 1.0
num_words = 1000

with open(model_data_filepath + 'out.txt', 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(num_words):
            output, hidden = model(input_, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(str(word.encode('utf-8')) + ('\n' if i % 20 == 19 else ' '))

            if i % 100 == 0:
                print('| Generated {}/{} words'.format(i, 1000))

with open(model_data_filepath + 'out.txt', 'r') as outf:
    all_output = outf.read()
    print(f'测试模型输出： {all_output}')

######################################################################
# 帮助函数

bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1


# create test data set
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into ``bsz`` parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the ``bsz`` batches.
    return data.view(bsz, -1).t().contiguous()


test_data = batchify(corpus.test, eval_batch_size)


# Evaluation functions
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(model_, data_source):
    # Turn on evaluation mode which disables dropout.
    model_.eval()
    total_loss = 0.
    hidden = model_.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model_(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


######################################################################
# 4. 测试动态量化大小
# ----------------------------
# 需要设置量化后端，不然会报错
torch.backends.quantized.engine = 'qnnpack'

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(f'量化后模型结构：{quantized_model}')


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


print('原始模型大小：')
print_size_of_model(model)
print('量化后模型大小：')
print_size_of_model(quantized_model)

######################################################################
# 测试推理速度
# torch.set_num_threads(1)


def time_model_evaluation(model, test_data):
    s = time.time()
    loss = evaluate(model, test_data)
    elapsed = time.time() - s
    print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))


print('原始模型推理速度：')
time_model_evaluation(model, test_data)
print('量化后模型推理速度：')
time_model_evaluation(quantized_model, test_data)



