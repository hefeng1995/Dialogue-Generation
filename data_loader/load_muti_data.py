from torchtext import data, datasets
import pickle
print('daily!!')


def padding_context(data, pad_index):
    dialog_max_len = max(list(map(lambda x: len(x), data)))
    for item in data:
        while(len(item) < dialog_max_len):
            item.append([pad_index])
    return data


def padding_sentence(data, max_len, pad_index):
    for sentence in data:
        while(len(sentence) < max_len):
            sentence.append(pad_index)
    return data


def data_iter(data, args, tensor_device):
    # shuffle data
    context, decoder_data, target_len = data
    for i in range(0, len(data) - args.batch_size, args.batch_size):
        context_batch = padding_context(
            context[i:i + args.batch_size], en_vocab.stoi['<pad>'])
        context_max_sentence = max(
            list(map(lambda x: len(x[0]), context_batch)))
        context_batch = list(map(lambda x: padding_sentence(
            x, context_max_sentence, en_vocab.stoi['<pad>']), context_batch))

        decoder_data_batch = padding_sentence(
            decoder_data[i:i + args.batch_size],
            max(list(map(lambda x: len(x),
                         decoder_data[i:i + args.batch_size]))),
            de_vocab.stoi['<pad>'])
        target_len_batch = target_len[i:i + args.batch_size]
        yield torch.tensor(context_batch).cuda(), torch.tensor(decoder_data_batch).cuda(),\
            torch.tensor(target_len_batch).cuda()


def load(data):
    context_data = pickle.load(open('pad_context_num.pkl', 'rb'))
    decoder_data = pickle.load(open('answer_num.pkl', 'rb'))
    de_vocab = pickle.load(open('de_vocab.pkl', 'rb'))
    en_vocab = pickle.load(open('en_vocab.pkl', 'rb'))
    return train_iter, val_iter, test_iter, en_vocab, de_vocab
