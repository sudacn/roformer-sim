import json 
import pandas as pd 
import numpy as np
import ipdb
from bert4keras.snippets import text_segmentate
# 基本信息
maxlen = 64
batch_size = 96
steps_per_epoch = 1000
epochs = 10000

# bert配置
config_path = '/root/ft_local/bert/chinese_roformer-sim-char_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/ft_local/bert/chinese_roformer-sim-char_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/ft_local/bert/chinese_roformer-sim-char_L-12_H-768_A-12/vocab.txt'
train_path = '/root/Gen-model/user_data/tmp_data/test_train.json'
csv_path = '/root/ft_local/model_data/hospital_sample.csv'

def load():
    f1 = pd.read_csv(csv_path)
    # for i in range
    return f1

def split(text):
    """分割句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen * 1.2, seps, strips)

def corpus():
    """读取语料
    """
    # ipdb.set_trace()
    d = load()
    # f2 = read('/root/data_pretrain/synonym_answers_shuf.json')
    # f3 = read('/root/data_pretrain/synonym/synonym_gen_2_shuf.json')
    
    for line in range(d.shape[0]):
        # d = next(f1)
        # ipdb.set_trace()
        text, synonyms = d['pname'][line], d['query_list'][line].split(";")
        text, synonym = np.random.permutation([text] + synonyms)[:2]
        text, synonym = split(text)[0], split(synonym)[0]
        yield text, synonym
        # d = next(f2)
        # text, synonym = d['text_a'], d['text_b']
        # text, synonym = split(text)[0], split(synonym)[0]
        # yield text, synonym
        # d = next(f1)
        # text, synonyms = d['text'], d['synonyms']
        # text, synonym = np.random.permutation([text] + synonyms)[:2]
        # text, synonym = split(text)[0], split(synonym)[0]
        # yield text, synonym
for i in corpus():
    print(i)

print("hello")
