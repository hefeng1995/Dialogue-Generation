import torch


def beamSearchDecoder(self, input_variable):
    decoder_hidden = encoder_hidden
    decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs)
    topk = decoder_output.data.topk(self.top_k)
    samples = [[] for i in range(self.top_k)]
    dead_k = 0
    final_samples = []
    for index in range(self.top_k):
        topk_prob = topk[0][0][index]
        topk_index = int(topk[1][0][index])
        samples[index] = [[topk_index], topk_prob, 0, 0, decoder_context,
                          decoder_hidden, decoder_attention, encoder_outputs]

    for _ in range(self.max_length):
        tmp = []
        for index in range(len(samples)):
            tmp.extend(self.beamSearchInfer(samples[index], index))
        samples = []

        # 筛选出topk
        df = pd.DataFrame(tmp)
        df.columns = ['sequence', 'pre_socres', 'fin_scores', "ave_scores",
                      "decoder_context", "decoder_hidden", "decoder_attention", "encoder_outputs"]
        sequence_len = df.sequence.apply(lambda x: len(x))
        df['ave_scores'] = df['fin_scores'] / sequence_len
        df = df.sort_values('ave_scores', ascending=False).reset_index().drop(
            ['index'], axis=1)
        df = df[:(self.top_k - dead_k)]
        for index in range(len(df)):
            group = df.ix[index]
            if group.tolist()[0][-1] == 1:
                final_samples.append(group.tolist())
                df = df.drop([index], axis=0)
                dead_k += 1
                print("drop {}, {}".format(group.tolist()[0], dead_k))
        samples = df.values.tolist()
        if len(samples) == 0:
            break

    if len(final_samples) < self.top_k:
        final_samples.extend(samples[:(self.top_k - dead_k)])
    return final_samples


def beamSearchInfer(self, sample, k):
    samples = []
    decoder_input = Variable(torch.LongTensor([[sample[0][-1]]]))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
    sequence, pre_scores, fin_scores, ave_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs = sample
    decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs)

    # choose topk
    topk = decoder_output.data.topk(self.top_k)
    for k in range(self.top_k):
        topk_prob = topk[0][0][k]
        topk_index = int(topk[1][0][k])
        pre_scores += topk_prob
        fin_scores = pre_scores - (k - 1) * self.alpha
        samples.append([sequence + [topk_index], pre_scores, fin_scores, ave_scores,
                        decoder_context, decoder_hidden, decoder_attention, encoder_outputs])
    return samples


def retrain(self):
    try:
        os.remove(self.model_path)
    except Exception as e:
        pass
    self.train()
