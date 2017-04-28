# coding=utf-8
from gensim import corpora, models, similarities
import jieba
import json
import sys
from pymongo import *
import io
import numpy as np
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print(os.getcwd())

client = MongoClient("localhost",27017)
db = client.oracledb
coll = db.oracle

print("reading sents...")
sentences = []
for it in coll.find().limit(100):
    sentences.append(it['content'])
print("sents:>> ", sentences)
#

#read stopwords
stopkeys = [line.strip() for line in open('stopwords.txt',encoding='utf-8').readlines()]


print("cutting...")
words = []# list<list>
for doc in sentences:
    words.append([word for word in list(jieba.cut(doc)) if word not in stopkeys and word !=' '])
print("ok.")
print(words[:10])

print("creating dict...")
dic=corpora.Dictionary(words)
dic.filter_extremes(no_below=1, no_above=0.8)# 去高频词
print("ok.")
print(dic)




print("creating corpus...")
corpus = [dic.doc2bow(text) for text in words]
print("ok.")

print("creating lda models...")
lda = models.ldamodel.LdaModel(corpus = corpus, id2word = dic, num_topics = 10, alpha = 'auto')
print("ok.")


#topics = lda.print_topics(10)


topics_matrix = lda.show_topics(formatted=True, num_words=10)

for topic in topics_matrix:
    print(str(topic))
#topics_matrix = np.array(topics_matrix)

