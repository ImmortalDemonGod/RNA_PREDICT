# rna_predict/pipeline/stageA/RFold_code.py
import logging
import os
import os.path as osp
import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


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
    message = ""
    for k, v in configs.items():
        message += "\n" + k + ": \t" + str(v) + "\t"
    return message


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


seq_dict = {"A": 0, "U": 1, "C": 2, "G": 3}


def base_matrix(_max_length, device):
    base_matrix = torch.ones(_max_length, _max_length)
    for i in range(_max_length):
        st, en = max(i - 3, 0), min(i + 3, _max_length - 1)
        for j in range(st, en + 1):
            base_matrix[i, j] = 0.0
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


def get_cut_len(length):
    return (((length - 1) // 16) + 1) * 16


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

    for i, pair_item in enumerate(pairs):
        col5[pair_item[0] - 1] = int(pair_item[1])
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack(
        (
            np.char.mod("%d", col1),
            col2,
            np.char.mod("%d", col3),
            np.char.mod("%d", col4),
            np.char.mod("%d", col5),
            np.char.mod("%d", col6),
        )
    ).T
    np.savetxt(
        osp.join(save_result_path, seq_name.replace("/", "_")) + ".ct",
        (temp),
        delimiter="\t",
        fmt="%s",
        header=">seq length: "
        + str(len(seq))
        + "\t seq name: "
        + seq_name.replace("/", "_"),
        comments="",
    )
    return


def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(["_"] * len(seq))
    dot_file[seq > idx] = "("
    dot_file[seq < idx] = ")"
    dot_file[seq == 0] = "."
    dot_file = "".join(dot_file)
    return dot_file


def save_ct(predict_matrix, seq_ori, name):
    seq_tmp = (
        torch.mul(
            predict_matrix.cpu().argmax(axis=1),
            predict_matrix.cpu().sum(axis=1).clamp_max(1),
        )
        .numpy()
        .astype(int)
    )
    seq_tmp[predict_matrix.cpu().sum(axis=1) == 0] = -1
    dot_list = seq2dot((seq_tmp + 1).squeeze())
    print(dot_list)
    letter = "AUCG"
    seq_letter = "".join([letter[item] for item in np.nonzero(seq_ori)[:, 1]])
    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    cur_pred = [
        (seq[0][i], seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0
    ]
    ct_file_output(cur_pred, seq_letter, name, "./")
    return


def visual_get_bases(seq):
    base_map = {"A": [], "U": [], "C": [], "G": []}
    for ii, s in enumerate(seq):
        if s in base_map:
            base_map[s].append(ii + 1)

    def to_comma_str(lst):
        return ",".join(str(x) for x in lst)

    a_bases = to_comma_str(base_map["A"])
    u_bases = to_comma_str(base_map["U"])
    c_bases = to_comma_str(base_map["C"])
    g_bases = to_comma_str(base_map["G"])
    return a_bases, u_bases, c_bases, g_bases


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, residual=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            return x + self.conv(x)
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum("... d, h d -> ... h d", x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class Attn(nn.Module):
    def __init__(self, *, dim, query_key_dim=128, expansion_factor=2.0, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qk = nn.Sequential(nn.Linear(dim, query_key_dim), nn.SiLU())
        self.offsetscale = OffsetScale(query_key_dim, heads=2)

    def forward(self, x):
        seq_len = x.shape[-2]
        normed_x = self.norm(x)
        qk = self.to_qk(normed_x)
        q, k = self.offsetscale(qk)
        sim = einsum("b i d, b j d -> b i j", q, k) / seq_len
        attn = F.relu(sim) ** 2
        return attn


class Encoder(nn.Module):
    def __init__(self, C_lst=[17, 32, 64, 128, 256]):
        super(Encoder, self).__init__()
        self.enc = nn.ModuleList([conv_block(ch_in=C_lst[0], ch_out=C_lst[1])])
        for ch_in, ch_out in zip(C_lst[1:-1], C_lst[2:]):
            self.enc.append(
                nn.Sequential(
                    *[
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        conv_block(ch_in=ch_in, ch_out=ch_out),
                    ]
                )
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
                nn.ModuleList(
                    [
                        up_conv(ch_in=ch_in, ch_out=ch_out),
                        conv_block(ch_in=ch_out * 2, ch_out=ch_out),
                    ]
                )
            )

    def forward(self, x, skips):
        skips.reverse()
        for i in range(0, len(self.dec)):
            upsample, conv = self.dec[i]
            x = upsample(x)
            x = conv(torch.cat((x, skips[i]), dim=1))
        return x


class Seq2Map(nn.Module):
    def __init__(
        self,
        input_dim=4,
        num_hidden=128,
        dropout=0.1,
        **kwargs,
    ):
        device = kwargs.pop("device", torch.device("cuda"))
        max_length = kwargs.pop("max_length", 3000)
        super(Seq2Map, self).__init__()
        self.device = device
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([num_hidden])).to(self.device)

        self.tok_embedding = nn.Embedding(input_dim, num_hidden)
        self.pos_embedding = nn.Embedding(self.max_length, num_hidden)
        self.layer = Attn(dim=num_hidden, query_key_dim=num_hidden, dropout=dropout)

    def forward(self, src):
        batch_size, src_len = src.shape[:2]
        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )
        src = self.tok_embedding(src) * self.scale
        src = self.dropout(src + self.pos_embedding(pos))
        attention = self.layer(src)
        return attention


class RFoldModel(nn.Module):
    def __init__(self, args):
        super(RFoldModel, self).__init__()

        c_in, c_out, c_hid = 1, 1, 32
        C_lst_enc = [c_in, 32, 64, 128, 256, 512]
        C_lst_dec = [2 * x for x in reversed(C_lst_enc[1:-1])] + [c_hid]

        self.encoder = Encoder(C_lst=C_lst_enc)
        self.decoder = Decoder(C_lst=C_lst_dec)
        self.readout = nn.Conv2d(c_hid, c_out, kernel_size=1, stride=1, padding=0)
        # Determine device from args (and fallback if torch.cuda.is_available is False)
        device_val = torch.device(
            "cuda"
            if getattr(args, "use_gpu", True) and torch.cuda.is_available()
            else "cpu"
        )
        self.seq2map = Seq2Map(
            input_dim=4,
            num_hidden=args.num_hidden,
            dropout=args.dropout,
            device=device_val,
        )

    def forward(self, seqs):
        attention = self.seq2map(seqs)
        x = (attention * torch.sigmoid(attention)).unsqueeze(0)
        latent, skips = self.encoder(x)
        latent = self.decoder(latent, skips)
        y = self.readout(latent).squeeze(1)
        return torch.transpose(y, -1, -2) * y
