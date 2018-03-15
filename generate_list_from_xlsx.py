#!/usr/bin/python

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
    cougroup = '''cpugroup="1"'''
    item = '''<item name="package_name" '''+ cougroup + ">" + package + '''</item>\n'''
    return item


def cross_xml(package):
    lines = open('device_policy.xml','r',errors='ignore').readlines()
    for line in lines:
        if package in line:
            print(package)
            return True
    return False

def process_game(data):
    table = data.sheet_by_name(u'game')
    nrows = table.nrows
    for i in range(nrows):
        package = table.cell(i, 0).value
        result = cross_xml(package)
        if result == False:
            item = get_cpugroup_item(package)
            write_to_file('whitelist_game.txt', item)
            #print(item)

def process_video(data):
    table = data.sheet_by_name(u'video')
    nrows = table.nrows
    for i in range(nrows):
        package = table.cell(i, 2).value
        item = get_memc_item(package)
        write_to_file('whitelist_video.txt', item)
        print(item)

def process_common(data):
    table = data.sheet_by_name(u'gallery')
    nrows = table.nrows
    for i in range(nrows):
        package = table.cell(i, 2).value
        item = get_common_item(package)
        write_to_file('whitelist_gallery.txt', item)
        print(item)

def process_xls():
    data = xlrd.open_workbook('111111.xlsx')
    process_game(data)
    #process_video(data)
    #process_common(data)

if __name__ == "__main__":
    imp.reload(sys)
    #sys.setdefaultencoding('utf-8')
    process_xls()
    