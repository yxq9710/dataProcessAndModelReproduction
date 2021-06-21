"""
构造词表：
    1.将训练集的所有数据都放在一个列表list中
    2.遍历列表list, 以每个word为key。出现频率为value构造一个字典
    3.较好那个字典按出现频率排序
    4.在字典中插入OOV的unused索引
"""

import nltk
from nltk.corpus import stopwords
import re  # 使用正则表达式处理unicode编码字符串

# nltk.download('punkt')
# nltk.download('stopwords')

word = set(stopwords.words('english'))
print(word)

dir1 = 'plot.tok.gt9.5000'
dir2 = 'quote.tok.gt9.5000'
dir3 = 'vocab_nltk.txt'


def build_vocab(dir1='plot.tok.gt9.5000', dir2='quote.tok.gt9.5000', dir3='vocab_nltk.txt', encode='ISO-8859-1',
                line_num=None):
    vocab_dict = {}
    line_list = []
    # 检查某行句子的分词情况
    # line_num = 4282
    for dir in [dir1, dir2]:
        if dir is None:  # 提高代码的泛化能力，增强可复用性
            continue
        with open(dir, 'r', encoding=encode) as f:
            i_num = 0
            lines = f.readlines()
            for line in lines:
                i_num += 1
                if line_num is not None:
                    if line_num == i_num:
                        print("到了指定的行：")
                line.lower().strip('\n').strip()
                line = re.sub(r"[^a-z?.!,]+", " ", line)  # 将除这些以外的全部替换为 “ ”
                line = nltk.tokenize.word_tokenize(line)

                # line_list = [w for w in line if w not in stopwords.words('english')]  # 去除停用词
                for w in line:
                    if w not in word:    # set(stopwords.words('english'))
                        line_list.append(w)

            for line_word in line_list:
                if line_word not in vocab_dict.keys():
                    vocab_dict[line_word] = 1
                else:
                    vocab_dict[line_word] += 1
            f.close()

    vocabs = sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)  # 设置为降序排列 ==》 得到 tuple(key, value)
    id = 1
    vocab_txt = {}
    for key, value in vocabs:
        # vocab_dict[key] = id   # 不能这样， 因为这样只是改变了key所对应的值， 而没有改变key之间的对应顺序
        vocab_txt[key] = id
        id += 1
    vocab_size = len(vocab_dict)
    print(vocab_size)

    with open(dir3, 'w', encoding='ISO-8859-1')as f:
        for key, value in vocab_txt.items():
            f.writelines(key + '\t' + str(value) + '\n')
        f.close()
    return vocab_txt


if __name__ == '__main__':
    vocabulary = build_vocab()
