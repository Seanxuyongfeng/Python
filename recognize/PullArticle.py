# -*- coding: utf8 -*-
#!/usr/bin/python

import urllib
import sys
from urllib import parse
import requests
import re
import os

TitleHeaders = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
    "Accept-Encoding": "text",
    'Accept-Language':u'zh-CN,zh;q=0.8,en;q=0.6,zh-TW;q=0.4',
    "Host":"apps.game.qq.com",
}

ArticleHeaders = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
    "Accept-Encoding": "text",
    'Accept-Language':u'zh-CN,zh;q=0.8,en;q=0.6,zh-TW;q=0.4',
    "Host":"apps.game.qq.com",
}

def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()  # entity全称，如&gt;
        key = sz.group('name')  # 去除&;后entity,如&gt;为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            # 以空串代替
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr

def handleHtml(htmlstr):
    # 先过滤CDATA
    re_backslash=re.compile(r'\\')
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)  # 匹配CDATA
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # style
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    re_comment = re.compile('<!--[^>]*-->')  # HTML注释
    s = re_backslash.sub('',htmlstr);
    s = re_cdata.sub('', s)  # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    s = re_br.sub('\n', s)  # 将br转换为换行
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_comment.sub('', s)  # 去掉HTML注释
    # 去掉多余的空行
    blank_line = re.compile('[ \n\t\f\v]+')
    s = blank_line.sub('\n', s)
    s = blank_line.sub('\n', s)
    s = replaceCharEntity(s)  # 替换实体
    return s



def downloadTitle(name):
    actionSearchUrl = u'http://apps.game.qq.com/wmp/v3.1/?p0=18&p1=searchIso&p2={word}&p3=NEWS&page={page}&pagesize=15&order=sIdxTime&r1=searchObj&openId=&agent=&channel=&area=&&_=1504773915672'
    session = requests.Session()
    session.headers = TitleHeaders
    referer = u'http://pvp.qq.com/web201605/searchResult.shtml?G_biz=18&sKeyword={keyword}'
    referer = referer.format(keyword=urllib.parse.quote(urllib.parse.quote(name)))
    session.headers['Referer'] = referer
    fileHandle = open('titles.txt', 'a+')
    fileHandle.write(name+'\n')
    for page in range(1000):
        print("try to get page "+str(page))
        html = session.get(actionSearchUrl.format(word=urllib.parse.quote(name.encode(encoding='gb2312',errors='ignore')),page=page), timeout=15)
        re_status = re.compile(r'"status":"(.*?)"')
        response_code = re_status.findall(html.content.decode())[0]
        if response_code == '-1' :
            break;
            print("Error")
        re_content = re.compile(r'"sTitle":"(.*?)".*?"iNewsId":"(.*?)"')
        datas = re_content.findall(html.content.decode().encode('utf-8').decode('unicode_escape'))
        for data in datas:
            print(data)
            fileHandle.write(data[0]+"\n"+data[1]+'\n')
            downloadArticle(name, data[1])

    fileHandle.close()

def downloadArticle(name, index):
    actionUrl = u'http://apps.game.qq.com/wmp/v3.1/public/searchNews.php?source=web_news_go&p0=18&id={index}&openId=&&agent=&&channel=&&area=&'
    session = requests.Session()
    session.headers = ArticleHeaders
    referer = u'http://pvp.qq.com/web201605/newsDetail.shtml?G_Biz=18&tid={index}'
    referer = referer.format(index=index)
    session.headers['Referer'] = referer
    
    html = session.get(actionUrl.format(index=index),timeout=15)
    re_content = re.compile(r'"sContent":"([.\S\s]*?)"')
    datas = re_content.findall(html.content.decode().encode('utf-8').decode('unicode_escape'))
    if len(datas) > 0:
        datas = handleHtml(datas[0])
        datas = handleHtml(datas)
        dirtName = os.getcwd() + "/article"
        print(dirtName)
        result = os.path.exists(dirtName)
        if result == False:
            os.mkdir(dirtName)

        fileHandle = open('article/article_'+name+'_'+index+'.txt', 'a+')
        fileHandle.write( datas + '\n')
        fileHandle.close()
    else:
        print("%s , %s has no sContent" % (name,index))

nameSet = set()
nameSet.add("廉颇")
nameSet.add("小乔")
nameSet.add("赵云")
nameSet.add("墨子")
nameSet.add("妲己")
nameSet.add("嬴政")
nameSet.add("孙尚香")
nameSet.add("鲁班七号")
nameSet.add("庄周")
nameSet.add("刘禅")
nameSet.add("高渐离")
nameSet.add("阿轲")
nameSet.add("钟无艳")
nameSet.add("孙膑")
nameSet.add("扁鹊")
nameSet.add("白起")
nameSet.add("芈月")
nameSet.add("吕布")
nameSet.add("周瑜")
nameSet.add("夏侯惇")
nameSet.add("甄姬")
nameSet.add("曹操")
nameSet.add("典韦")
nameSet.add("宫本武藏")
nameSet.add("李白")
nameSet.add("马可波罗")
nameSet.add("狄仁杰")
nameSet.add("达摩")
nameSet.add("项羽")
nameSet.add("武则天")
nameSet.add("老夫子")
nameSet.add("关羽")
nameSet.add("貂蝉")
nameSet.add("安琪拉")
nameSet.add("程咬金")
nameSet.add("露娜")
nameSet.add("姜子牙")
nameSet.add("刘邦")
nameSet.add("韩信")
nameSet.add("王昭君")
nameSet.add("兰陵王")
nameSet.add("花木兰")
nameSet.add("张良")
nameSet.add("不知火舞")
nameSet.add("娜可露露")
nameSet.add("橘右京")
nameSet.add("亚瑟")
nameSet.add("孙悟空")
nameSet.add("牛魔")
nameSet.add("后羿")
nameSet.add("刘备")
nameSet.add("张飞")
nameSet.add("李元芳")
nameSet.add("虞姬")
nameSet.add("钟馗")
nameSet.add("成吉思汗")
nameSet.add("杨戬")
nameSet.add("雅典娜")
nameSet.add("蔡文姬")
nameSet.add("太乙真人")
nameSet.add("哪吒")
nameSet.add("诸葛亮")
nameSet.add("黄忠")
nameSet.add("大乔")
nameSet.add("东皇太一")
nameSet.add("干将莫邪")
nameSet.add("鬼谷子")
nameSet.add("铠")
nameSet.add("百里守约")
nameSet.add("百里玄策")

for name in nameSet:
    downloadTitle(name)