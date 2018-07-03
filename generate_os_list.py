# -*- coding: utf-8 -*-
#!/usr/bin/python
#python2.7
#whitelist.xlsx
import xlrd
import string
import sys
import imp


def write_to_file(filename, msg):
    file = open(filename,'a')
    file.write(msg)
    file.close()

def get_memc_item(package):
    memc = '''memc="1"'''
    item = '''<item name="package_name" '''+ memc + ">" + package + '''</item>\n'''
    return item

def get_common_item(package):
    item = '''<item name="package_name">'''+ package + '''</item>\n'''
    return item
    
def get_cpugroup_item(package):
    cougroup = '''cpugroup="16777228"'''
    item = '''<item name="package_name" '''+ cougroup + ">" + package + '''</item>\n'''
    return item

#open('device_policy.xml','r',errors='ignore').readlines()
def cross_xml(package):
    lines = open('device_policy.xml').readlines()
    for line in lines:
        if package in line:
            print("repeat:" + package)
            return True
    return False

def process_game(data):
    table = data.sheet_by_name(u'game')
    nrows = table.nrows
    for i in range(nrows):
        package = table.cell(i, 1).value
        result = cross_xml(package)
        if result == False:
            item = get_cpugroup_item(package)
            write_to_file('whitelist_game.xml', item)
            print(item)

def process_video(data):
    table = data.sheet_by_name(u'video_memc')
    nrows = table.nrows
    for i in range(nrows):
        package = table.cell(i, 1).value
        result = cross_xml(package)
        if result == False:
            item = get_memc_item(package)
            write_to_file('video_memc.txt', item)
            print(item)

def process_common(data):
    table = data.sheet_by_name(u'video_common')
    nrows = table.nrows
    for i in range(nrows):
        package = table.cell(i, 1).value
        result = cross_xml(package)
        if result == False:
            item = get_common_item(package)
            write_to_file('vodeo_common.xml', item)
            #print(item)

def process_xls():
    data = xlrd.open_workbook('os_list.xlsx')
    #process_game(data)
    process_video(data)
    #process_common(data)

if __name__ == "__main__":
    imp.reload(sys)
    sys.setdefaultencoding('utf8')
    if sys.version_info.major != 2 and sys.version_info.minor != 7:
        print("need python 2.7 but get " + sys.version_info)
    else :
        process_xls()
