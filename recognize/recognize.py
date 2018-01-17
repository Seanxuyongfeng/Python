# -*- coding: utf8 -*-
#!/usr/bin/python
import os
import re
from collections import Counter
#import numpy as np
import tflearn

OUTPUT_DIR = u'./output'

heroNameList = list()
heroWordsList = list()

heroNameList.clear()
heroWordsList.clear()

for fileName in os.listdir(OUTPUT_DIR):
    re_name= re.compile(r'_(.*)_')
    #get hero name
    name = re_name.findall(fileName)
    wordsFile = open(OUTPUT_DIR+u'/'+fileName,'r')
    heroWordsList.append(wordsFile.readlines())
    if(len(name) > 0):
        heroNameList.append(name[0])
    else:
        print("error = "+fileName)

print(len(heroNameList))
print(len(heroWordsList))

totalWords = Counter()

totalWords.clear()
# get all the words in splited article
for article in heroWordsList:
    for word in article:
        word = word.strip()
        totalWords[word]+=1

print('total words:', len(totalWords))

# get the very first 2000 words
selectedWords = sorted(totalWords, key=totalWords.get,reverse=True)[:10000]
print(selectedWords[:20])

# query words by index
word2idx = {word: i for i, word in enumerate(selectedWords)}

# describe splited article by index
def text_to_vector(splitedArticle):
    # empty int vector
    word_vector = np.zeros(len(vocab), dtype=np.int_)
    for word in splitedArticle:
        word = word.strip()
        index = word2idx.get(word,None)
        if index is None:
            continue
        else:
            word_vector[index]+=1
    
    return np.array(word_vector)

#print(text_to_vector(heroWordsList[100]))

# remove the duplicated hero names
nameSet = set()
for name in heroNameList:
    nameSet.add(name)
print(len(nameSet))

# get all the hero names,26
hero2index = list()
for name in nameSet:
    hero2index.append(name)

#save the hero index in heroIndexList
heroIndexList = list()
for name in heroNameList:
    heroIndexList.append(hero2index.index(name))


