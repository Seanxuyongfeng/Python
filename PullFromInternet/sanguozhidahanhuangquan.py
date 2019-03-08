# -*- coding: utf8 -*-
#!/usr/bin/python
import requests
import re
from html.parser import HTMLParser

TitleHeaders = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
    "Accept-Encoding": "text",
    'Accept-Language':u'zh-CN,zh;q=0.8,en;q=0.6,zh-TW;q=0.4',
}

class MULUHTMLParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.data = []
        self.flag = False

    def getData(self):
        return self.data

    def handle_starttag(self, tag, attrs):
        """
        recognize start tag, like <div>
        :param tag:
        :param attrs:
        :return:
        """

        #print("Encountered a start tag:", tag)
        #print("tag",tag)
        #print(attrs)
        try:
            for attr in attrs:
                #print(attr)
                #print(attr[1])
                #print(attr[2])
                if attr[1].endswith('.html'):
                    #if self.flag == True:
                    self.flag == True
                    self.data.append('https://www.i7wx.com/book/70/70402/'+str(attr[1]))
                elif attr[1] == 'a':
                    self.flag = True
        except AttributeError as e:
            pass
        except IndexError as e1:
            pass
    def handle_endtag(self, tag):
        """
        recognize end tag, like </div>
        :param tag:
        :return:
        """
        if tag == 'div' and self.flag == True:
            self.flag = False
        #print("Encountered an end tag :", tag)

    def handle_data(self, data):
        """
        recognize data, html content string
        :param data:
        :return:
        """
        #print("Encountered some data  :", data)

    def handle_startendtag(self, tag, attrs):
        """
        recognize tag that without endtag, like <img />
        :param tag:
        :param attrs:
        :return:
        """
        #print("Encountered startendtag :", tag)

    def handle_comment(self,data):
        """
        :param data:
        :return:
        """
        #print("Encountered comment :", data)


class ArticalHTMLParser(HTMLParser):

    def save_to_file(self, content):
        fileHandle = open('zhengwen.txt', 'a+',encoding='ISO-8859-1')
        fileHandle.write(content)
        fileHandle.close()

    def __init__(self):
        HTMLParser.__init__(self)
        self.curtag = ""
        self.data = []

    def handle_starttag(self, tag, attrs):
        """
        recognize start tag, like <div>
        :param tag:
        :param attrs:
        :return:
        """
        #self.save_to_file("Encountered a start tag:")
        #self.save_to_file(tag)
        self.curtag = tag
        for attr in attrs:
            if attr[1] == 'content':
                self.curtag = 'content'


    def handle_endtag(self, tag):
        """
        recognize end tag, like </div>
        :param tag:
        :return:
        """
        #self.save_to_file("Encountered an end tag :")
        #self.save_to_file(tag)
        #print("Encountered an end tag :", tag)
        pass

    def handle_data(self, data):
        """
        recognize data, html content string
        :param data:
        :return:
        """
        #self.save_to_file("Encountered some data  :")
        #self.save_to_file(data)
        #print("Encountered some data  :", data)
        if self.curtag == 'h1':
            self.save_to_file(data)
        elif self.curtag == 'content':
            re_br = re.compile('<br\s*?/?>')  # 处理换行
            s = re_br.sub('\n', data)  # 将br转换为换行
            blank_line = re.compile('[ \n\t\f\v]+')
            s = blank_line.sub('\n', s)
            s = blank_line.sub('\n', s)
            self.save_to_file(s)

    def handle_startendtag(self, tag, attrs):
        """
        recognize tag that without endtag, like <img />
        :param tag:
        :param attrs:
        :return:
        """
        #self.save_to_file("Encountered startendtag :")
        #self.save_to_file(tag)
        #print("Encountered startendtag :", tag)
        pass

    def handle_comment(self,data):
        """
        :param data:
        :return:
        """
        #self.save_to_file("Encountered comment :")
        #self.save_to_file(data)
        #print("Encountered comment :", data)
        pass

 
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

session = requests.Session()
session.headers = TitleHeaders

muluhtml = session.get("https://www.i7wx.com/book/70/70402")
#fileHandle = open('aaaaaa.txt', 'a+',encoding='utf-8')
fileHandle = open('bbbbbbbbbbb.txt', 'a+',encoding='ISO-8859-1')
fileHandle.write(muluhtml.text)
fileHandle.close()
muluparse = MULUHTMLParser()
muluparse.feed(muluhtml.text)
mulu = muluparse.getData()

for mm in mulu:
    print("-----------")
    print(mm)
    articals = session.get(mm)
    #print(articals.text)
    parse = ArticalHTMLParser()
    sss = replaceCharEntity(articals.text);
    parse.feed(sss);



#html.encoding = 'utf-8'
#fileHandle = open('aaaaaa.txt', 'a+',encoding='utf-8')
#fileHandle = open('bbbbbbbbbbb.txt', 'a+',encoding='ISO-8859-1')
#fileHandle.write(html.text)
#fileHandle.close()
#parser = MULUHTMLParser()
#parser.feed(html.text);
#print(html.encoding)


#zhangjie = session.get("https://www.88dushu.com/xiaoshuo/26/26122/5476978.html")
#parse = ArticalHTMLParser()
#sss = replaceCharEntity(zhangjie.text);
#parse.feed(sss)