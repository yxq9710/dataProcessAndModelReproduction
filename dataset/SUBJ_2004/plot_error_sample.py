from dataset.SUBJ_2004.utils import *

vocab_size = 23906
embedding_dim = 300
max_len = 64
model = position_attention(vocab_size=vocab_size, output_dim=1, embedding_dim=embedding_dim, max_len=max_len)

vocab = get_vocab(dir1='plot.tok.gt9.5000', dir2='quote.tok.gt9.5000', dir3='vocab.txt')

i = 9
checkpoint_filepath = 'errors_4/checkpoint/checkpoint'
filepath = checkpoint_filepath + '_' + str(i)
model.load_weights(filepath)
model.compile(optimizer=optimizers.Adam(),  # 静态学习率没有动态学习率更适应模型
              loss=losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

sentences = get_error_sentence(dir='errors_4/error_sample_' + str(i) + '.txt')
sentences = pad_list_word(sentences, vocab, max_len)
probability = np.zeros([len(sentences), 1])
for id, sentence in enumerate(sentences):
    probability[id] = Evaluate_error(model, sentence, vocab, max_len)

# modify_false_sample(i, probability)
print()


"""
  生成"_extract.txt"文件 
"""
# if __name__ == '__main__':
#     for i in range(10):
#         modify_false_sample(i)

