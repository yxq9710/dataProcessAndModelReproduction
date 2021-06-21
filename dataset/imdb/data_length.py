import tensorflow as tf
import pandas as pd


def get_median(data):
    """
    获得中位数

    arguments:
        data: 一个包含所有数据的列表
    step1: 列表排序
    step2：找到列表长度lengths，得到data[lengths/2]
    """
    data = sorted(data)
    lengths = len(data)
    if lengths % 2 == 0:
        median_position = tf.cast(lengths / 2, tf.int32)
        num = (data[median_position] + data[median_position+1]) / 2
    else:
        median_position = tf.cast((lengths-1) / 2, tf.int32)
        num = data[median_position]
    return num


def get_frequency_digit(data):
    """
    获取列表中的众数
    arguments：
        data: 一个包含上所有数据的列表
    """
    frequency_digit = max(data, key=lambda v: data.count(v))
    not_frequency_digit = min(data, key=lambda v: data.count(v))
    return frequency_digit, not_frequency_digit


imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

a = set()
c = []
sum = 0
for i in range(len(x_train)):
    lens = len(x_train[i])
    sum += lens
    c.append(lens)
    a.add(lens)

"""
符号的意义：
    b_max:       最大值
    b_min:       最小智
    b_median:    中位数
    b_mean:      平均数
    b_frequency: 众数     c.count(b_frequency) ： 众数的频数
"""

b_max = max(a)
b_min = min(a)
b_median = get_median(c)
b_mean = sum/(i+1)
b_frequency, b_not_frequency = get_frequency_digit(c)
print(c.count(b_frequency))
print(c.count(b_not_frequency))
print(b_max)
print(sum/(i+1))
print()

dict1 = {'name:': ['imdb_test'],
         '最大值': [b_max],
         '最小值': [b_min],
         '平均数': [b_mean],
         '中位数': [b_median],
         '众数' : [b_frequency],
         '众数的频数': [c.count(b_frequency)]}
df = pd.DataFrame(dict1)
name = 'data.xlsx'
# df.to_excel('data.xlsx')
with pd.ExcelWriter(name, mode='a') as writer:
    df.to_excel(writer)

# dic1 = {'标题列1': ['张三','李四'],
#         '标题列2': [80, 90]
#        }
# df = pd.DataFrame(dic1)
# df.to_excel('1.xlsx', index=False)
