# coding=utf-8
# 输入: txt, 每行一个文本
import os
from gensim import corpora, models, similarities
import jieba
print(os.getcwd())


import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json

infile = 'in/weixin_samp.txt'
outfile = 'out/matrix.txt'

stopkeys = [line.strip() for line in open('in/stopwords.txt', encoding='utf-8').readlines()]
# print(stopkeys)

print('cutting...')

docs=[]
file = open(infile,mode='r', encoding='utf-8')
for doc in file:
    # print(doc)
    docs.append(doc)
file.close()
words = []
for doc in docs:
    words.append([word for word in list(jieba.cut(doc)) if word not in stopkeys and word !=' '])

# print(words[:10])
print("creating dict...")
dic=corpora.Dictionary(words)
print(dic)


print("creating corpus...")
corpus = [dic.doc2bow(text) for text in words]
print("ok.")


print("creating lda models...")
lda = models.ldamodel.LdaModel(corpus = corpus, id2word = dic, num_topics = 10, alpha = 'auto')
print("ok.")

topics_matrix = lda.show_topics(formatted=True, num_words=10)

writer = open(outfile,'w',encoding='utf-8')
for topic in topics_matrix:
    print(str(topic))
    writer.write(str(topic)+'\r\n')
writer.close()


print("all done.")