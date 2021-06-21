"""
    对原始样本数据进行处理，去除了一些停用词
"""

import nltk
from nltk import WordNetLemmatizer as word_net
from nltk.corpus import stopwords
import re
from dataset.SUBJ_2004.build_vocabulary import build_vocab

dir1 = 'plot.tok.gt9.5000'
dir2 = 'quote.tok.gt9.5000'


word = set(stopwords.words('english'))
print(word)

line_num = None


for dir in [dir1, dir2]:
    line_two = []

    """
    放在代码的末尾
    if dir is dir1:
        line1 = line_two
    else:
        line2 = line_two
        
    """
    if dir is None:  # 提高代码的泛化能力，增强可复用性
        continue
    with open(dir, 'r', encoding='ISO-8859-1') as f:
        i_num = 0
        lines = f.readlines()
        for line in lines:
            line_list = []
            i_num += 1
            if line_num is not None:
                if line_num == i_num:
                    print("到了指定的行：")
            line.lower().strip('\n').strip()
            line = re.sub(r"[^a-z?.!,0-9]+", " ", line)  # 将除这些以外的全部替换为 “ ”
            line = nltk.tokenize.word_tokenize(line)

            # for w in line:
            #     if w not in word:  # set(stopwords.words('english'))
            #         w = word_net().lemmatize(w)     # 在去除停用词的同时进行词性还原
            #         line_list.append(w)

            # for w in line:
            #     w = word_net().lemmatize(w)
            #     line_list.append(w)
            line_list = line
            line_str = " ".join(line_word for line_word in line_list) + '\n'
            line_two.append(line_str)
        f.close()
    with open('processed_' + dir, 'w', encoding="ISO-8859-1") as f:
        for line in line_two:
            f.writelines(line)
        f.close()


vocabulary = build_vocab(dir1='processed_plot.tok.gt9.5000', dir2='processed_quote.tok.gt9.5000', dir3='vocab_nltk_re.txt')
