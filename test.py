# coding=utf-8
from gensim import corpora, models, similarities

import jieba
import json

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 输入
sentences = ["我喜欢吃土豆","土豆是个百搭的东西","我不喜欢今天雾霾的北京"]

# 分词
words=[]
for doc in sentences:
    words.append(list(jieba.cut(doc)))
#print(words)

# 构造词典
dic=corpora.Dictionary(words)
print(dic)
print(dic.token2id)

# 生成语料，词频统计,转化成空间向量格式
corpus = [dic.doc2bow(text) for text in words]
print("corpus>> ", corpus)


lda = models.ldamodel.LdaModel(corpus=corpus,id2word=dic,num_topics=3,alpha='auto')
for pattern in lda.show_topics():
    print( "%s" % str(pattern))
