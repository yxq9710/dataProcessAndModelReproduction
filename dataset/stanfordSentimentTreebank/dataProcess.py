# Copyright 2018 lww. All Rights Reserved.
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


def delblankline(infile1, infile2, trainfile, validfile, testfile):
    with open(infile1, 'r') as info1, open(infile2, 'r') as info2, \
            open(trainfile, 'w') as train, open(validfile, 'w') as valid, open(testfile, 'w') as test:
        lines1 = info1.readlines()
        lines2 = info2.readlines()
        for i in range(1, len(lines1)):
            t1 = lines1[i].replace("-LRB-", "(")
            t2 = t1.replace("-RRB-", ")")
            k = lines2[i].strip().split(",")
            t = t2.strip().split('\t')
            if k[1] == '1':
                train.writelines(t[1])
                train.writelines("\n")
            elif k[1] == '2':
                test.writelines(t[1])
                test.writelines("\n")
            elif k[1] == '3':
                valid.writelines(t[1])
                valid.writelines("\n")
        print("end")


def tag_sentiment(infile, infile0, infile1, infile2):
    # ("sentiment_labels.txt", "dictionary.txt", "train.txt","train_final.txt")
    with open(infile, 'r') as info, open(infile0, 'r', encoding='utf-8') as info0, open(infile1, 'r', encoding='utf-8') as info1, \
            open(infile2, 'w', encoding='utf-8') as info2:
        lines = info.readlines()
        lines0 = info0.readlines()
        lines1 = info1.readlines()

        # text2id = {}
        # for i in range(0, len(lines0)):
        #     s = lines0[i].strip().split("|")
        #     text2id[s[0]] = s[1]

        id2sentiment = {}
        for i in range(0, len(lines)):
            s = lines[i].strip().split("|")
            id2sentiment[s[0]] = s[1]

        for line in lines1:
            line = line.strip()
            for i in range(0, len(lines0)):
                s = lines0[i].strip().split("|")
                if s[0] == line:
                    text_id = s[1]
                    break
            # if line.strip() not in text2id:
            #     print(line.strip())
            #     # 由于特殊字符不匹配造成
            #     continue
            # else:
            #     text_id = text2id[line.strip()]
            sentiment_score = id2sentiment[text_id]
            info2.write(line.strip() + "\t" + str(sentiment_score) + "\n")
        print("end3d1")


# delblankline("datasetSentences.txt", "datasetSplit.txt", "train.txt", "valid.txt", "test.txt")
# 获取原始的训练集，测试集，验证集

"""
    已处理，每次处理train.txt时耗时甚巨，所以没有必要，就不要再处理了
"""
# train有8544条，dev有1101条，test有 2210条
# tag_sentiment("sentiment_labels.txt", "dictionary.txt", "train.txt", "train_final.txt")
# tag_sentiment("sentiment_labels.txt", "dictionary.txt", "valid.txt", "valid_final.txt")
# tag_sentiment("sentiment_labels.txt", "dictionary.txt", "test.txt", "test_final.txt")

