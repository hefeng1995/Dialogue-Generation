from torchtext import data, datasets
import pickle
print('daily!!')


def load_data(args, tensor_device):
    def de_tokenizer(text):
        temp = text.split(' ')
        return temp

    def en_tokenizer(text):
        temp = text.split(' ')
        return temp
    en = data.Field(
        sequential=True, tokenize=en_tokenizer, lower=True, include_lengths=True)
    de = data.Field(
        sequential=True, include_lengths=True, lower=True, tokenize=de_tokenizer, init_token='<go>', eos_token='<eos>')

    train = data.Dataset(pickle.load(open('./data/daily_train_ex.pkl', 'rb')),
                         fields=[('de', de), ('en', en)])
    test = data.Dataset(pickle.load(open('./data/daily_test_ex.pkl', 'rb')),
                        fields=[('de', de), ('en', en)])
    val = data.Dataset(pickle.load(open('./data/daily_test_ex.pkl', 'rb')),
                       fields=[('de', de), ('en', en)])
    en.build_vocab(train, vectors='glove.6B.300d')
    de.build_vocab(train, min_freq=2)
    train_iter = data.BucketIterator(
        dataset=train, batch_size=args.batch_size,
        sort=False, train=True, shuffle=True, sort_within_batch=True,
        sort_key=lambda x: len(x.en),
        device=tensor_device, repeat=False)
    val_iter = data.BucketIterator(
        dataset=val, batch_size=args.batch_size,
        sort=False,  sort_within_batch=True,
        sort_key=lambda x: len(x.en),
        device=tensor_device, repeat=False)
    test_iter = data.BucketIterator(
        dataset=test,  batch_size=args.batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.en),
        device=tensor_device, repeat=False)

    return train_iter, val_iter, test_iter, en, de
