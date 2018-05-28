from torchtext import data, datasets
import pickle
print('lm!!')


def load_data(args, tensor_device):
    def en_tokenizer(text):
        temp = text.split(' ')
        return temp
    en = data.Field(
        sequential=True, tokenize=en_tokenizer, lower=True, include_lengths=True, init_token='<go>', eos_token='<eos>')
    train = data.TabularDataset(
        'ptb.train.csv', 'csv', fields=[('en', en)])
    val = data.TabularDataset(
        'ptb.valid.csv', 'csv', fields=[('en', en)])
    test = data.TabularDataset(
        'ptb.valid.csv', 'csv', fields=[('en', en)])
    en.build_vocab(train, val, test, vectors='glove.6B.300d')
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

    return train_iter, val_iter, test_iter, en
