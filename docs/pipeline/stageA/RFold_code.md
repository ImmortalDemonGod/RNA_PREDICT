RFold CODE:
Directory Structure:

└── ./
    ├── API
    │   ├── __init__.py
    │   ├── dataloader.py
    │   ├── dataset.py
    │   └── metric.py
    ├── colab_utils.py
    ├── main.py
    ├── model.py
    ├── module.py
    ├── parser.py
    ├── rfold.py
    └── utils.py



---
File: /API/__init__.py
---

from .dataloader import load_data
from .metric import evaluate_result


---
File: /API/dataloader.py
---

from .dataset import RNADataset
from torch.utils.data import DataLoader


def load_data(data_name, batch_size, data_root, num_workers=8, **kwargs):
    if data_name == 'RNAStralign':
        test_set = RNADataset(path=data_root, dataname='test_600')
    elif data_name == 'ArchiveII':
        test_set = RNADataset(path=data_root, dataname='all_600')
    elif data_name == 'bpRNA':
        test_set = RNADataset(path=data_root, dataname='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return test_loader


---
File: /API/dataset.py
---

import numpy as np
import os.path as osp
import _pickle as cPickle
from tqdm import tqdm
from torch.utils import data


def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l


class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)
        return attr


class RNADataset(data.Dataset):
    def __init__(self, path, dataname):
        self.path = path
        self.dataname = dataname
        self.data = self.cache_data

    def __len__(self):
        return len(self.data)

    def get_data(self, dataname):
        filename = dataname + '.pickle'
        pre_data = cPickle.load(open(osp.join(self.path, filename), 'rb'))

        data = []
        for instance in tqdm(pre_data):
            data_x, _, seq_length, name, pairs = instance
            l = get_cut_len(seq_length, 80)
            # contact
            contact = np.zeros((l, l))
            contact[tuple(np.transpose(pairs))] = 1. if pairs != [] else 0.
            # data_seq
            data_seq = np.zeros((l, 4))
            data_seq[:seq_length] = data_x[:seq_length]
            data.append([contact, seq_length, data_seq])
        return data

    @cached_property
    def cache_data(self):
        return self.get_data(self.dataname)

    def __getitem__(self, index):
        return self.data[index]


---
File: /API/metric.py
---

import torch


def evaluate_result(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a) * torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps)/(tp+fn+eps)
    precision = (tp + eps)/(tp+fp+eps)
    f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
    return precision, recall, f1_score


---
File: /colab_utils.py
---

import torch
import os.path as osp
import numpy as np
import torch.nn.functional as F


seq_dict = {
    'A': 0,
    'U': 1,
    'C': 2,
    'G': 3
}


def base_matrix(_max_length, device):
    base_matrix = torch.ones(_max_length, _max_length)
    for i in range(_max_length):
        st, en = max(i-3, 0), min(i+3, _max_length-1)
        for j in range(st, en + 1):
            base_matrix[i, j] = 0.
    return base_matrix.to(device)

def constraint_matrix(x):
    base_a, base_u, base_c, base_g = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return (au_ua + cg_gc + ug_gu) * base_matrix(x.shape[1], x.device)

def sequence2onehot(seq, device):
    seqs = list(map(lambda x: seq_dict[x], seq))
    return torch.tensor(seqs).unsqueeze(0).to(device)

def get_cut_len(l):
    return (((l - 1) // 16) + 1) * 16

def process_seqs(seq, device):
    seq_len = len(seq)
    seq = sequence2onehot(seq, device=device)
    nseq_len = get_cut_len(seq_len)
    nseq = F.pad(seq, (0, nseq_len - seq_len))
    nseq_one_hot = F.one_hot(nseq).float()
    return nseq, nseq_one_hot, seq_len
  
def row_col_softmax(y):
    row_softmax = torch.softmax(y, dim=-1)
    col_softmax = torch.softmax(y, dim=-2)
    return 0.5 * (row_softmax + col_softmax)

def row_col_argmax(y):
    y_pred = row_col_softmax(y)
    y_hat = y_pred + torch.randn_like(y) * 1e-12
    col_max = torch.argmax(y_hat, 1)
    col_one = torch.zeros_like(y_hat).scatter(1, col_max.unsqueeze(1), 1.0)
    row_max = torch.argmax(y_hat, 2)
    row_one = torch.zeros_like(y_hat).scatter(2, row_max.unsqueeze(2), 1.0)
    int_one = row_one * col_one 
    return int_one

def ct_file_output(pairs, seq, seq_name, save_result_path):
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]-1] = int(I[1])
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    np.savetxt(osp.join(save_result_path, seq_name.replace('/','_'))+'.ct', (temp), delimiter='\t', fmt="%s", header='>seq length: ' + str(len(seq)) + '\t seq name: ' + seq_name.replace('/','_') , comments='')
    return

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def save_ct(predict_matrix, seq_ori, name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    dot_list = seq2dot((seq_tmp+1).squeeze())
    letter = 'AUCG'
    seq_letter = ''.join([letter[item] for item in np.nonzero(seq_ori)[:,1]])
    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    cur_pred = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    ct_file_output(cur_pred, seq_letter, name, './')
    return 

def visual_get_bases(seq):
    a_bases, u_bases, c_bases, g_bases = [], [], [], []
    for ii, s in enumerate(seq):
        if s == 'A': a_bases.append(ii+1)
        if s == 'U': u_bases.append(ii+1)
        if s == 'C': c_bases.append(ii+1)
        if s == 'G': g_bases.append(ii+1)
    a_bases = ''.join([str(s)+',' for s in a_bases])[:-1]
    u_bases = ''.join([str(s)+',' for s in u_bases])[:-1]
    c_bases = ''.join([str(s)+',' for s in c_bases])[:-1]
    g_bases = ''.join([str(s)+',' for s in g_bases])[:-1]
    return a_bases, u_bases, c_bases, g_bases


---
File: /main.py
---

import json
import torch
import logging
import collections
import os.path as osp
# from parser import create_parser

import warnings
warnings.filterwarnings('ignore')

from utils import *
from rfold import RFold


class Exp:
    def __init__(self, args):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()

    def _build_method(self):
        self.method = RFold(self.args, self.device)

    def _get_data(self):
        self.test_loader = get_dataset(self.config)

    def test(self):
        test_f1, test_precision, test_recall, test_runtime = self.method.test_one_epoch(self.test_loader)
        print_log('Test F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, Runtime: {3:.4f}\n'.format(test_f1, test_precision, test_recall, test_runtime))
        return test_f1, test_precision, test_recall, test_runtime

# if __name__ == '__main__':
#     RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

#     args = create_parser()
#     config = args.__dict__
#     exp = Exp(args)

#     print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#     exp.test()
#     print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#     test_f1, test_precision, test_recall, test_runtime = exp.test()


---
File: /model.py
---

import torch
import torch.nn as nn
from module import conv_block, up_conv, Attn


class Encoder(nn.Module):
    def __init__(self, C_lst=[17, 32, 64, 128, 256]):
        super(Encoder, self).__init__()
        self.enc = nn.ModuleList([conv_block(ch_in=C_lst[0],ch_out=C_lst[1])])
        for ch_in, ch_out in zip(C_lst[1:-1], C_lst[2:]):
            self.enc.append(
                nn.Sequential(*[
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    conv_block(ch_in=ch_in, ch_out=ch_out)
                ])
            )

    def forward(self, x):
        skips = []
        for i in range(0, len(self.enc)):
            x = self.enc[i](x)
            skips.append(x)
        return x, skips[:-1]


class Decoder(nn.Module):
    def __init__(self, C_lst=[512, 256, 128, 64, 32]):
        super(Decoder, self).__init__()
        self.dec = nn.ModuleList([])
        for ch_in, ch_out in zip(C_lst[0:-1], C_lst[1:]):
            self.dec.append(
                nn.ModuleList([
                    up_conv(ch_in=ch_in, ch_out=ch_out),
                    conv_block(ch_in=ch_out * 2, ch_out=ch_out)
                ])
            )

    def forward(self, x, skips):
        skips.reverse()
        for i in range(0, len(self.dec)):
            upsample, conv = self.dec[i]
            x = upsample(x)
            x = conv(torch.cat((x, skips[i]), dim=1))
        return x


class Seq2Map(nn.Module):
    def __init__(self, 
                 input_dim=4,
                 num_hidden=128,
                 dropout=0.1, 
                 device=torch.device('cuda'),
                 max_length=3000,
                 **kwargs):
        super(Seq2Map, self).__init__(**kwargs)
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([num_hidden])).to(device)
        
        self.tok_embedding = nn.Embedding(input_dim, num_hidden)
        self.pos_embedding = nn.Embedding(max_length, num_hidden)
        self.layer = Attn(dim=num_hidden, query_key_dim=num_hidden, dropout=dropout)

    def forward(self, src):
        batch_size, src_len = src.shape[:2]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.tok_embedding(src) * self.scale
        src = self.dropout(src + self.pos_embedding(pos))
        attention = self.layer(src)
        return attention

    
class RFold_Model(nn.Module):
    def __init__(self, args):
        super(RFold_Model, self).__init__()

        c_in, c_out, c_hid = 1, 1, 32
        C_lst_enc = [c_in, 32, 64, 128, 256, 512]
        C_lst_dec = [2*x for x in reversed(C_lst_enc[1:-1])] + [c_hid]

        self.encoder = Encoder(C_lst=C_lst_enc)
        self.decoder = Decoder(C_lst=C_lst_dec)
        self.readout = nn.Conv2d(c_hid, c_out, kernel_size=1, stride=1, padding=0)
        self.seq2map = Seq2Map(input_dim=4, num_hidden=args.num_hidden, dropout=args.dropout)

    def forward(self, seqs):
        attention = self.seq2map(seqs)
        x = (attention * torch.sigmoid(attention)).unsqueeze(0)
        latent, skips = self.encoder(x)
        latent = self.decoder(latent, skips)
        y = self.readout(latent).squeeze(1)
        return torch.transpose(y, -1, -2) * y


---
File: /module.py
---

import torch
from torch import nn, einsum
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,residual=False):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.residual = residual

    def forward(self,x):
        if self.residual:
            return x + self.conv(x)
        return self.conv(x)
        

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)


class Attn(nn.Module):
    def __init__(self, *, dim, query_key_dim=128, expansion_factor=2.,
        dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )
        self.offsetscale = OffsetScale(query_key_dim, heads=2)

    def forward(self, x):
        seq_len = x.shape[-2]
        normed_x = self.norm(x)
        qk = self.to_qk(normed_x)
        q, k = self.offsetscale(qk)
        sim = einsum('b i d, b j d -> b i j', q, k) / seq_len
        attn = F.relu(sim) ** 2
        return attn


---
File: /parser.py
---

import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)

    # dataset parameters
    parser.add_argument('--data_name', default='ArchiveII', choices=['ArchiveII', 'RNAStralign', 'bpRNA'])
    parser.add_argument('--data_root', default='./data/archiveII_all')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    # Training parameters
    parser.add_argument('--epoch', default=1, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # debug parameters
    parser.add_argument('--num_hidden', default=128, type=int)
    parser.add_argument('--pf_dim', default=128, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    return parser.parse_args()


---
File: /rfold.py
---

import time
import torch
import numpy as np
from tqdm import tqdm
from utils import cuda
from API import evaluate_result
from model import RFold_Model


# predefine a base_matrix
_max_length = 1005
base_matrix = torch.ones(_max_length, _max_length)
for i in range(_max_length):
    st, en = max(i-3, 0), min(i+3, _max_length-1)
    for j in range(st, en + 1):
        base_matrix[i, j] = 0.

def constraint_matrix(x):
    base_a, base_u, base_c, base_g = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + torch.transpose(au, -1, -2)
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + torch.transpose(ug, -1, -2)
    return (au_ua + cg_gc + ug_gu) * base_matrix[:length, :length].to(x.device)

def row_col_softmax(y):
    row_softmax = torch.softmax(y, dim=-1)
    col_softmax = torch.softmax(y, dim=-2)
    return 0.5 * (row_softmax + col_softmax)

def row_col_argmax(y):
    y_pred = row_col_softmax(y)
    y_hat = y_pred + torch.randn_like(y) * 1e-12
    col_max = torch.argmax(y_hat, 1)
    col_one = torch.zeros_like(y_hat).scatter(1, col_max.unsqueeze(1), 1.0)
    row_max = torch.argmax(y_hat, 2)
    row_one = torch.zeros_like(y_hat).scatter(2, row_max.unsqueeze(2), 1.0)
    int_one = row_one * col_one 
    return int_one


class RFold(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.config = args.__dict__

        self.model = self._build_model()
        self.criterion = torch.nn.MSELoss()

    def _build_model(self, **kwargs):
        return RFold_Model(self.args).to(self.device)

    def test_one_epoch(self, test_loader, **kwargs):
        # note that the model is under the training mode for bn/dropout
        self.model.train()
        eval_results, run_time = [], []
        test_pbar = tqdm(test_loader)
        for batch in test_pbar:
            contacts, seq_lens, seq_ori = batch
            contacts, seq_ori = cuda(
                (contacts.float(), seq_ori.float()), device=self.device)

            # predict
            seqs = torch.argmax(seq_ori, axis=-1)
            s_time = time.time()
            with torch.no_grad():
                pred_contacts = self.model(seqs)

            pred_contacts = row_col_argmax(pred_contacts) * constraint_matrix(seq_ori)

            # interval time
            interval_t = time.time() - s_time
            run_time.append(interval_t)

            eval_result = list(map(lambda i: evaluate_result(pred_contacts.cpu()[i],
                                                                     contacts.cpu()[i]), range(contacts.shape[0])))
            eval_results += eval_result

        p, r, f1 = zip(*eval_results)
        return np.average(f1), np.average(p), np.average(r), np.average(run_time)


---
File: /utils.py
---

import os
import logging
import numpy as np
import torch
from torch import optim
import random 
import torch.backends.cudnn as cudnn
from collections.abc import Mapping, Sequence


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_dataset(config):
    from API import load_data
    return load_data(**config)

def cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, *args, **kwargs)
    raise TypeError("Can't transfer object type `%s`" % type(obj))
