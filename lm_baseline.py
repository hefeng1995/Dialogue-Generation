import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sumeval.metrics.bleu import BLEUCalculator
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from cuda_functional import SRU, SRUCell
from data_loader.load_dialog_data import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print('118!')


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=80, type=int,
                        help='epoch')
    parser.add_argument('--device', default=1, type=int,
                        help='gpu')
    parser.add_argument('--topk', default=5,
                        type=int, help='topk')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    parser.add_argument('--beam_size', default=5, type=int,
                        help='beam size for beam search')
    parser.add_argument('--sample_size', default=10,
                        type=int, help='sample size')
    parser.add_argument('--embedding_size', default=300,
                        type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=512,
                        type=int, help='size of hidden states')
    parser.add_argument('--layer_size', default=1,
                        type=int, help='size of layers')
    parser.add_argument('--input_drop_out', default=0.2,
                        type=float, help='dropout rate for rnn input')
    parser.add_argument('--output_drop_out', default=0.2,
                        type=float, help='dropout rate for rnn output')
    parser.add_argument('--drop_out', default=0.2,
                        type=float, help='dropout rate')
    parser.add_argument('--rnn_cell', default='GRU',
                        type=str, help='GRU cell')
    parser.add_argument('--decode_max_time_step', default=20, type=int, help='maximum number of time steps used '
                        'in decoding and sampling')
    parser.add_argument('--grad_clip', default=5.,
                        type=float, help='clip gradients')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='decay learning rate if the validation performance drops')
    parser.add_argument('--bidirectional', default=True, type=bool,
                        help='is bidirectional')
    parser.add_argument('--z_dim', default=16, type=int,
                        help='dimension of latent variable')
    args = parser.parse_args()
    return args


class LM(nn.Module):
    def __init__(self, args, en_vocab):
        super(LM, self).__init__()
        self.args = args
        self.input_drop = nn.Dropout(p=args.input_drop_out)
        self.encoder_embedding = nn.Embedding(
            len(en_vocab), args.embedding_size, padding_idx=en_vocab.stoi['<pad>'])
        self.linear_vocab = nn.Linear(
            args.hidden_size * args.layer_size, len(en_vocab))
        self.encoder_embedding.weight.data.copy_(en_vocab.vectors)
        self.rnn = nn.GRU(input_size=args.embedding_size,
                          hidden_size=args.hidden_size,
                          num_layers=args.layer_size,
                          bias=True,
                          dropout=args.drop_out,
                          )

    def forward(self, padded_encoder_inputs):
        encoder_embedded = self.encoder_embedding(padded_encoder_inputs)
        encoder_embedded = self.input_drop(encoder_embedded)
        encoder_output, _ = self.rnn(encoder_embedded)
        logits = self.linear_vocab(encoder_output)
        return logits


def train(epoch, model, optimizer, train_iter, en, de, args, test_iter=None):
    total_loss = 0
    max_bleu = 0.0001
    pad = en.vocab.stoi['<pad>']
    de_vocab_size = len(de.vocab)
    max_epoch = 0
    for epoch in range(1, epoch + 1):
        for step, batch in enumerate(train_iter):
            model.train()
            decoder_data, decoder_length = batch.en
            if(decoder_data.size(1) != 64):
                continue
            decoder_inputs = decoder_data[0:-1, :]
            target = decoder_data[1:, :]
            logits = model(decoder_inputs)
            optimizer.zero_grad()
            loss = F.cross_entropy(logits.view(-1, de_vocab_size),
                                   target.contiguous(
            ).view(-1),
                ignore_index=pad)
            clip_grad_norm_(model.parameters(), args.grad_clip)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            if step % 300 == 0 and step != 0:
                total_loss = total_loss / 300
                print('---------------------------------')
                print("[%d][%d][loss:%5.6f][pp:%5.6f]" %
                      (epoch, step, total_loss, math.exp(total_loss)))
                eval(model, test_iter, en, de)
                total_loss = 0
        epoch_loss = eval(model, test_iter, en, de, epoch_end=True)


def eval(model, data_iter, en, de, epoch_end=False):
    en_vocab_size = len(en.vocab)
    pad = en.vocab.stoi['<pad>']
    model.eval()
    if(epoch_end):
        total_loss = []
        for step, batch in enumerate(data_iter):
            decoder_data, _ = batch.en
            decoder_inputs = decoder_data[0:-1, :]
            target = decoder_data[1:, :]
            if(decoder_inputs.size(1) != 64):
                continue
            logits = model(decoder_inputs)
            loss = F.cross_entropy(logits.view(-1, en_vocab_size),
                                   target.contiguous(
            ).view(-1),
                ignore_index=pad)
            total_loss.append(loss)
        return np.mean(total_loss)


def main():
    args = init_config()
    print('[!]Loading and preparing dataset...')

    if torch.cuda.is_available():
        tensor_device = 0
    else:
        tensor_device = -1
    train_iter, val_iter, test_iter, en, de = load_data(args, tensor_device)
    print('[!]Done...')
    en_size, de_size = len(en.vocab), len(de.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[EN_vocab]:%d [DE_vocab]:%d" % (en_size, de_size))
    print("[!] Instantiating models...")
    _go = torch.tensor([de.vocab.stoi['<go>']]
                       ).expand(args.batch_size)
    if torch.cuda.is_available():
        _go = _go.cuda()
    encoder = Encoder(args, en.vocab)
    decoder = Decoder(args, de.vocab)
    seq2seq = Seq2Seq(args, encoder, decoder, _go)
    print(seq2seq)
    print('Number of the model :')
    print(sum([param.nelement() for param in seq2seq.parameters()]))
    if torch.cuda.is_available():
        seq2seq = seq2seq.cuda()
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr)
    # for e in range(1, args.epoch + 1):
    train(args.epoch, seq2seq, optimizer, train_iter, en, de, args, test_iter)

    # val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN)
    # val_loss = 0
    # print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
    #        % (e, val_loss, math.exp(val_loss)))


main()
