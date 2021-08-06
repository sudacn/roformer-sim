#! -*- coding: utf-8 -*-
# SimBERT v2 训练代码
# 训练环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.6

import json
from operator import is_
import ipdb
import sys
from ipdb.__main__ import main
import numpy as np
import pandas as pd 
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import text_segmentate, truncate_sequences
from bert4keras.snippets import AutoRegressiveDecoder
import jieba
from pandas.io.parsers import read_csv
jieba.initialize()
sys.path.append('/root/gitWOA/roformer-sim')
from config import Config


# 建立分词器
tokenizer = Tokenizer(Config.dict_path, do_lower_case=True)

def load():
    f1 = pd.read_csv(Config.train_path)
    # for i in range
    return f1

def split(text):
    """分割句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, Config.maxlen * 1.2, seps, strips)

def corpus():
    """读取语料
    """
    d = load()
    for line in range(d.shape[0]):
        # d = next(f1)
        # ipdb.set_trace() 
        text, address, synonyms = d['pname'][line], d['address'][line], d['query_list'][line].split(";")
        for i in synonyms[:10]:
            text, synonym = text, np.random.permutation(synonyms[:5])[0]
            text, synonym = split(text)[0], split(synonym)[0]
            yield text+"-"+address, synonym
        # d = next(f2)
        # text, synonym = d['text_a'], d['text_b']
        # text, synonym = split(text)[0], split(synonym)[0]
        # yield text, synonym
        # d = next(f1)
        # text, synonyms = d['text'], d['synonyms']
        # text, synonym = np.random.permutation([text] + synonyms)[:2]
        # text, synonym = split(text)[0], split(synonym)[0]
        # yield text, synonym
        # d = next(f3)
        # text, synonym = d['text_a'], d['text_b']
        # text, synonym = split(text)[0], split(synonym)[0]
        # yield text, synonym

def corpus_test():
    """
    读取语料
    """
    d = read_csv(Config.test_path)
    texts = []
    for line in range(d.shape[0]):
        # d = next(f1)
        # ipdb.set_trace()  
        text, address, synonyms = d['pname'][line], d['address'][line], d['query_list'][line].split(";")
        texts.append(text+"-"+address)
    return texts

def masked_encode(text):
    """wwm随机mask
    """
    words = jieba.lcut(text)
    rands = np.random.random(len(words))
    source, target = [tokenizer._token_start_id], [0]
    for r, w in zip(rands, words):
        ids = tokenizer.encode(w)[0][1:-1]
        if r < 0.15 * 0.8:
            source.extend([tokenizer._token_mask_id] * len(ids))
            target.extend(ids)
        elif r < 0.15 * 0.9:
            source.extend(ids)
            target.extend(ids)
        elif r < 0.15:
            source.extend(
                np.random.choice(tokenizer._vocab_size - 1, size=len(ids)) + 1
            )
            target.extend(ids)
        else:
            source.extend(ids)
            target.extend([0] * len(ids))
    source = source[:Config.maxlen - 1] + [tokenizer._token_end_id]
    target = target[:Config.maxlen - 1] + [0]
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.some_samples = []

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (text, synonym) in self.sample(random):
            for i in range(1): # 单向 1  双向 2 
                if np.random.random() < 0.5:
                    text_ids = masked_encode(text)[0]
                else:
                    text_ids = tokenizer.encode(text)[0]
                synonym_ids = tokenizer.encode(synonym)[0][1:]
                truncate_sequences(Config.maxlen * 2, -2, text_ids, synonym_ids)
                token_ids = text_ids + synonym_ids
                segment_ids = [0] * len(text_ids) + [1] * len(synonym_ids)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                self.some_samples.append(text)
                if len(self.some_samples) > 1000:
                    self.some_samples.pop(0)
                # text, synonym = synonym, text
            if len(batch_token_ids) == Config.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class TotalLoss(Loss):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """
    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 20  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels


# 建立模型，加载模型权重
roformer = build_transformer_model(
    Config.config_path,
    Config.checkpoint_path,
    model='roformer',
    application='unilm',
    with_pool='linear',
    with_mlm='linear',
    dropout_rate=0.2,
    ignore_invalid_weights=True
)

encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])

outputs = TotalLoss([2, 3])(roformer.inputs + roformer.outputs)
model = keras.models.Model(roformer.inputs, outputs)

AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=1e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.summary()


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=Config.maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=Config.maxlen
)

def is_good_generator(synonym,text):
    # words = jieba.lcut(synonym)
    for word in synonym:
        if word not in text:
            return False
    return True 



def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    效果：
        >>> gen_synonyms(u'微信和支付宝哪个好？')
        [
            u'微信和支付宝，哪个好?',
            u'微信和支付宝哪个好',
            u'支付宝和微信哪个好',
            u'支付宝和微信哪个好啊',
            u'微信和支付宝那个好用？',
            u'微信和支付宝哪个好用',
            u'支付宝和微信那个更好',
            u'支付宝和微信哪个好用',
            u'微信和支付宝用起来哪个好？',
            u'微信和支付宝选哪个好',
        ]
    """
    text_list = text.split("-")
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i not in text_list and is_good_generator(i,text)]
    # r = sorted(r,key=lambda i : len(i),reverse=False)
    # print(r)
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    result =  [r[i + 1] for i in argsort[:k]]
    result = sorted(result, key = lambda i:len(i),reverse=False )
    return result

def just_show():
    """
    随机观察一些样本的效果
    """
    some_samples = train_generator.some_samples
    S = [np.random.choice(some_samples) for i in range(3)]
    for s in S:
        try:
            print(u'原句子：%s' % s)
            print(u'同义句子：')
            print(gen_synonyms(s, 10, 10))
            print()
        except:
            pass

def just_show_test():
    
    S = corpus_test()
    # print(S)
    for s in S:
        try:
            print(u'原句子：%s' % s)
            print(u'同义句子：')
            print(gen_synonyms(s, 10, 10))
            print()
        except:
            pass

class Evaluate(keras.callbacks.Callback):
    """评估模型
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights(Config.last_model_path)
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(Config.best_model_path)
        # 演示效果
        just_show()
            

if __name__ == '__main__':

    train_generator = data_generator(corpus(), Config.batch_size)
    # evaluator = Evaluate()
    # model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=Config.steps_per_epoch,
    #     epochs=Config.epochs,
    #     callbacks=[evaluator]
    # )
    model.load_weights(Config.best_model_path)
    just_show_test()

else:
    model.load_weights(Config.best_model_path)
    just_show_test()
