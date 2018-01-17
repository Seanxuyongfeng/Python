# -*- coding: utf8 -*-
#!/usr/bin/python

import requests
import re
import os
import codecs
import time

def splite_once(words,filename):
    postData = {
        "q":words,
        "fmt":"js"
    }
    session = requests.Session()
    session.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
            "Accept-Encoding": "gzip, deflate",
            'Accept-Language':u'zh-CN,zh;q=0.8,en;q=0.6,zh-TW;q=0.4',
            "Host":"www.sogou.com",
            "Accept":"application/json, text/javascript, */*; q=0.01",
            "Origin":"http://www.sogou.com",
            "Referer":"http://www.sogou.com/labs/webservice/",
            "X-Requested-With":"XMLHttpRequest",
        }
    url = u'http://www.sogou.com/labs/webservice/sogou_word_seg.php'
    html = session.post(url,data = postData)
    content = html.content.decode().encode('utf-8').decode('unicode_escape')
    re_result=re.compile(r'"result":\[(\[.*?\])\]')
    result = re_result.findall(content)
    re_word = re.compile(r'"(.*?)"')
    if(len(result) == 0):
        return
    textToken = re_word.findall(result[0])
    output_dir = 'output/'
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    output_file = open('output/'+filename,'a+')
    for token in textToken[::2]:
        output_file.writelines(token+"\n")
    output_file.close()

output_dir = os.getcwd() + "/split_output"

result = os.path.exists(output_dir)
if result == False:
    os.mkdir(output_dir)

article_path = 'article/'

articles = os.listdir(article_path)

for article in articles:
    if os.path.exists(article_path+article):
        print(article)
        file_handle = codecs.open(article_path+article, 'r','GBK')
        article_content = file_handle.read()
        article_len = len(article_content.strip())
        print('spliting... ' + article)
        #print('article len = ' + str(article_len))
        if article_len < 2000:
            continue
        splite_once_len = 2000
        count = 0
        while (count < article_len):
            start_index = count
            if start_index >= article_len:
                break

            end_index = start_index + splite_once_len
            if end_index >= article_len:
                end_index = article_len + 1 # rest of them

            #print('send once :'+article_content[start_index:end_index])
            splite_once(article_content[start_index:end_index],article)
            count += splite_once_len

        #print(article_content)
    else:
        print('File not exists')
    
    time.sleep(2)

print('Splite Done!')