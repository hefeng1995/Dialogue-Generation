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
from load_dialog_data import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print('yes??')


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=80, type=int,
                        help='epoch')
    parser.add_argument('--average_size', default=5, type=int,
                        help='compress the logits')
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
    parser.add_argument('--input_drop_out', default=0.1,
                        type=float, help='dropout rate for rnn input')
    parser.add_argument('--drop_out', default=0.1,
                        type=float, help='dropout rate')
    parser.add_argument('--output_drop_out', default=0.1,
                        type=float, help='dropout rate for rnn output')
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


def id2string(symbols, id2w_vocab):
    all_str = []
    for batch_index in range(symbols.size(0)):
        all_str.append([id2w_vocab[id] for id in symbols[batch_index]])
    return all_str


class Encoder(nn.Module):
    def __init__(self, args, en_vocab):
        super(Encoder, self).__init__()
        encoder_rnn_cell = 'SRU'
        self.dense_hidden = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.input_drop = nn.Dropout(p=args.input_drop_out)
        self.output_drop = nn.Dropout(p=args.output_drop_out)
        self.encoder_embedding = nn.Embedding(
            len(en_vocab), args.embedding_size, padding_idx=en_vocab.stoi['<pad>'])
        self.encoder_embedding.weight.data.copy_(en_vocab.vectors)
        if(encoder_rnn_cell == 'GRU'):
            self.rnn = nn.GRU(input_size=args.embedding_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.layer_size,
                              bias=True,
                              dropout=args.drop_out,
                              bidirectional=args.bidirectional
                              )
        elif(encoder_rnn_cell == 'LSTM'):
            self.rnn = nn.LSTM(input_size=args.embedding_size,
                               hidden_size=args.hidden_size,
                               num_layers=args.layer_size,
                               bias=True,
                               dropout=args.drop_out,
                               bidirectional=args.bidirectional
                               )
        elif(encoder_rnn_cell == 'SRU'):
            self.rnn = SRU(input_size=args.embedding_size,
                           hidden_size=args.hidden_size,
                           num_layers=args.layer_size,          # number of stacking RNN layers
                           dropout=args.drop_out,           # dropout applied between RNN layers
                           rnn_dropout=0.2,       # variational dropout applied on linear transformation
                           use_tanh=1,            # use tanh?
                           use_relu=0,            # use ReLU?
                           use_selu=0,            # use SeLU?
                           bidirectional=True,   # bidirectional RNN ?
                           weight_norm=False,     # apply weight normalization on parameters
                           layer_norm=False,      # apply layer normalization on the output of each layer
                           # initial bias of highway gate (<= 0)
                           highway_bias=0
                           )

    def forward(self, padded_encoder_inputs, encoder_input_len=0):
        encoder_embedded = self.encoder_embedding(padded_encoder_inputs)
        encoder_embedded = self.input_drop(encoder_embedded)
        # encoder_embedded = nn.utils.rnn.pack_padded_sequence(
        #    encoder_embedded, encoder_input_len)
        # zero init
        encoder_output, hidden_state = self.rnn(encoder_embedded)
        # print(hidden_state.shape)
        # print(hidden_state.shape)
        '''
        hidden_state = self.dense_hidden(
            torch.cat([hidden_state[0], hidden_state[1]], -1))
        '''
        # sru
        hidden_state = self.dense_hidden(hidden_state)
        #hidden_state = self.output_drop(hidden_state)
        # print('what??????')
        #print(str(e_time - s_time))
        return encoder_output, hidden_state
        # return encoder_output, hidden_state.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, args, de_vocab):
        super(Decoder, self).__init__()
        self.avg_num = args.average_size
        self.de_vocab = de_vocab
        self.args = args
        self.output_size = len(de_vocab)
        self.decoder_vocab_dense = nn.Linear(
            args.hidden_size, self.output_size)
        self.infer_vocab_dense = nn.Linear(args.hidden_size, self.output_size)
        self.input_drop = nn.Dropout(p=args.input_drop_out)
        self.output_drop = nn.Dropout(p=args.output_drop_out)
        self.decoder_embedding = nn.Embedding(
            len(de_vocab), args.embedding_size, padding_idx=de_vocab.stoi['<pad>'])
        if(args.rnn_cell == 'GRU'):
            self.decoder_rnn = nn.GRU(input_size=args.embedding_size,
                                      hidden_size=args.hidden_size,
                                      num_layers=args.layer_size,
                                      bias=True,
                                      dropout=args.drop_out
                                      )
            self.infer_rnn = nn.GRU(input_size=args.hidden_size,
                                    hidden_size=args.hidden_size,
                                    num_layers=args.layer_size,
                                    bias=True,
                                    dropout=args.drop_out
                                    )
        elif(args.rnn_cell == 'LSTM'):
            self.rnn = nn.LSTM(input_size=args.embedding_size,
                               hidden_size=args.hidden_size,
                               num_layers=args.layer_size,
                               bias=True,
                               dropout=args.drop_out
                               )
        elif(args.rnn_cell == 'SRU'):
            self.rnn = SRU(input_size=args.embedding_size,
                           hidden_size=args.hidden_size,
                           num_layers=6,          # number of stacking RNN layers
                           dropout=0.2,           # dropout applied between RNN layers
                           rnn_dropout=0.2,       # variational dropout applied on linear transformation
                           use_tanh=1,            # use tanh?
                           use_relu=0,            # use ReLU?
                           use_selu=0,            # use SeLU?
                           bidirectional=False,   # bidirectional RNN ?
                           weight_norm=False,     # apply weight normalization on parameters
                           layer_norm=False,      # apply layer normalization on the output of each layer
                           # initial bias of highway gate (<= 0)
                           highway_bias=0
                           )

    def forward(self, decoder_input, h_decoder, h_infer):
        decoder_input = self.input_drop(
            self.decoder_embedding(decoder_input).unsqueeze(0))
        #decoder_input = self.decoder_embedding(decoder_input).unsqueeze(0)
        decoder_output, decoder_hidden_state = self.decoder_rnn(
            decoder_input, h_decoder)
        decoder_output = self.output_drop(decoder_output)
        decoder_logits = self.decoder_vocab_dense(decoder_output)
        #decoder_prob = F.softmax(decoder_logits / 20, -1)
        infer_rnn_input = decoder_output
        infer_output, infer_hidden_state = self.infer_rnn(
            infer_rnn_input, h_infer)
        infer_output = self.output_drop(infer_output)
        infer_logits = self.infer_vocab_dense(infer_output)
        return decoder_logits, decoder_hidden_state, infer_logits, infer_hidden_state


class Seq2Seq(nn.Module):
    def __init__(self, args, encoder, decoder, _go, _unk):
        super(Seq2Seq, self).__init__()
        self.batch_size = args.batch_size
        self.encoder = encoder
        self.decoder = decoder
        self._unk = _unk
        self._go = _go
        self.word_drop = 0.20

    def forward(self, encoder_inputs, encoder_length, decoder_inputs, train_infer=True):
        _, encoder_final_state = self.encoder(encoder_inputs, encoder_length)
        max_len = decoder_inputs.size(0)
        decoder_logits = []
        infer_logits = []
        decoder_symbols = []
        infer_symbols = []
        decoder_hidden = encoder_final_state
        infer_hidden = encoder_final_state
        output = self._go
        for t in range(1, max_len + 1):
            decoder_logit, decoder_hidden, infer_logit, infer_hidden = self.decoder(
                output, decoder_hidden, infer_hidden)
            decoder_logits.append(decoder_logit)
            infer_logits.append(infer_logit)
            is_drop = random.random() < self.word_drop
            decoder_top1 = decoder_logit.data.argmax(-1)
            decoder_symbols.append(decoder_top1)
            infer_top1 = infer_logit.data.argmax(-1)
            infer_symbols.append(infer_top1)

            if(t < max_len):
                if(self.training and is_drop):
                    output = self._unk
                    continue
                if(train_infer):
                    output = decoder_inputs[t] if self.training else infer_top1.squeeze(
                    )
                else:
                    output = decoder_inputs[t] if self.training else decoder_top1.squeeze(
                    )
        return decoder_logits, decoder_symbols, infer_logits, infer_symbols


def train(epoch, model, optimizer, train_iter, en, de, args, test_iter=None):
    total_loss = 0
    max_bleu = 0.0001
    max_epoch = 0
    pad = en.vocab.stoi['<pad>']
    de_vocab_size = len(de.vocab)
    for epoch in range(1, epoch + 1):
        for step, batch in enumerate(train_iter):
            model.train()
            encoder_inputs, encoder_length = batch.en
            if(encoder_inputs.size(1) != 64):
                continue
            decoder_data, decoder_length = batch.de
            decoder_inputs = decoder_data[0:-1, :]
            # print(id2string(decoder_inputs.t(), de.vocab.itos)[0])
            target = decoder_data[1:, :]
            # print(id2string(target.t(), de.vocab.itos)[0])
            decoder_logits, decoder_symbols, infer_logits, infer_symbols = model(
                encoder_inputs, encoder_length, decoder_inputs)
            # print(len(decoder_logits))
            # print(target.shape)
            optimizer.zero_grad()
            decoder_loss = 0
            infer_loss = 0
            sentence_len = len(decoder_logits)
            for index in range(sentence_len):
                decoder_loss = decoder_loss + F.cross_entropy(decoder_logits[index].view(-1, de_vocab_size),
                                                              target[index].contiguous(
                ).view(-1), size_average=False,
                    ignore_index=pad)
                infer_loss = infer_loss + F.cross_entropy(infer_logits[index].view(-1, de_vocab_size),
                                                          target[index].contiguous(
                ).view(-1), size_average=False,
                    ignore_index=pad)
            ave_len = torch.sum(decoder_length.float())
            decoder_loss = decoder_loss / ave_len.float()
            infer_loss = infer_loss / ave_len.float()
            loss = infer_loss + decoder_loss
            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.data.item()
            if step % 100 == 0 and step != 0:
                total_loss = total_loss / 100
                print('---------------------------------')
                print("[%d][%d][total_step:%d][loss:%5.6f][pp:%5.6f]" %
                      (epoch, step, len(train_iter), total_loss, math.exp(total_loss)))
                print('[Max_bleu:%5.6f][Max_epoch:%d]' % (max_bleu, max_epoch))
                '''
                decoder_str = torch.cat(decoder_symbols, 0).t()
                print('---------------------------------')
                print('Encoder Decoder inputs and Targets')
                print(
                    ' '.join(id2string(encoder_inputs.t(), en.vocab.itos)[0]))
                print(
                    ' '.join(id2string(decoder_inputs.t(), de.vocab.itos)[0]))
                print(' '.join(id2string(target.t(), de.vocab.itos)[0]))
                print('---------------------------------')
                print('output')
                print(' '.join(id2string(decoder_str, de.vocab.itos)[0]))
                print('---------------------------------')
                '''
                eval(model, test_iter, en, de)
                total_loss = 0
        epoch_bleu = eval(model, test_iter, en, de, epoch_end=True)
        if(epoch_bleu > max_bleu):
            max_bleu = epoch_bleu
            max_epoch = epoch


def cal_bleu(prediction_str, target_str):
    bleu = BLEUCalculator()
    total_bleu = []
    for index in range(len(prediction_str)):
        prediction_rel = ' '.join(prediction_str[index])
        eos_index = prediction_rel.find('<eos>')
        if(eos_index > 0):
            prediction_rel = prediction_rel[:eos_index - 1]
        target_rel = ' '.join(target_str[index])
        target_rel = target_rel[:target_rel.find('<eos>') - 1]
        total_bleu.append(
            bleu.bleu(prediction_rel, target_rel))
    return np.mean(total_bleu)


def eval(model, data_iter, en, de, epoch_end=False):
    model.eval()
    if(epoch_end):
        total_bleu = []
        for step, batch in enumerate(data_iter):
            encoder_inputs, encoder_length = batch.en
            decoder_data, _ = batch.de
            decoder_inputs = decoder_data[0:-1, :]
            target = decoder_data[1:, :]
            if(encoder_inputs.size(1) != 64):
                continue
            decoder_logits, decoder_symbols, infer_logits, infer_symbols = model(
                encoder_inputs, encoder_length, decoder_inputs)
            infer_symbols = torch.cat(infer_symbols, 0).t()
            target_str = id2string(target.t(), de.vocab.itos)
            infer_str = id2string(infer_symbols, de.vocab.itos)
            total_bleu.append(cal_bleu(infer_str, target_str))
        return np.mean(total_bleu)
    else:
        for batch in data_iter:
            encoder_inputs, encoder_length = batch.en
            decoder_data, _ = batch.de
            decoder_inputs = decoder_data[0:-1, :]
            target = decoder_data[1:, :]
            if(encoder_inputs.size(1) != 64):
                return 0
            decoder_logits, decoder_symbols, infer_logits, infer_symbols = model(
                encoder_inputs, encoder_length, decoder_inputs)
            decoder_str = torch.cat(decoder_symbols, 0).t()
            infer_str = torch.cat(infer_symbols, 0).t()
            target_str = id2string(target.t(), de.vocab.itos)
            decoder_str = id2string(decoder_str, de.vocab.itos)
            infer_str = id2string(infer_str, de.vocab.itos)
            print('Bleu')
            print(cal_bleu(infer_str, target_str))
            print('--------------------------')
            print('Encoder Decoder inputs :')
            print(id2string(encoder_inputs.t(), en.vocab.itos)[0])
            print('Targets :')
            print(target_str[0])
            print('--------------------------')
            print('Eval decode output :')
            print(decoder_str[0])
            print('Eval infer output :')
            print(infer_str[0])
            print('--------------------------')
            break


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
    _unk = torch.tensor([de.vocab.stoi['<unk>']]
                        ).expand(args.batch_size)
    if torch.cuda.is_available():
        _go = _go.cuda()
        _unk = _unk.cuda()
    encoder = Encoder(args, en.vocab)
    decoder = Decoder(args, de.vocab)
    # seq2seq = Seq2Seq(args, encoder, decoder, _go, _unk)
    seq2seq = Seq2Seq(args, encoder, decoder, _go, _unk)
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
