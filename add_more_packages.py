# -*- coding: utf-8 -*-
#!/usr/bin/python
#python3.6
#whitelist.xlsx
import xlrd
import string
import sys
import imp
import re
import os

CLASS_INDEX = 1
ID_INDEX = 2
MEMC_INDEX = 3
PACKAGE_INDEX = 4
NEW_DEVICE_POLICY = 'device_policy_addition.xml'

def write_to_file(filename, msg):
    file = open(filename,'a',encoding='utf8')
    file.write(msg)
    file.close()

#open('device_policy.xml','r',errors='ignore').readlines()
def cross_xml(package):
    lines = open('device_policy.xml',encoding='utf8').readlines()
    for line in lines:
        pkgs = re.findall(".*>(.*)<.*", line)
        for content in pkgs:
            if package == content:
                print("repeat: " + content)
                return True
    return False

def cell_valid(cell):
    cell = str(cell)
    cell = cell.strip()
    if cell.strip() != '' and len(cell) != 0:
        return True
    else:
        return False

def memc_valid(memc):
    if memc != 0.0:
        memc = str(memc)
        memc = memc.strip()
        if memc.strip() != '' and len(memc) != 0:
            return True
        else:
            return False
    else:
        return False

def write_to_new_xml_header():
    global NEW_DEVICE_POLICY
    if os.path.exists(NEW_DEVICE_POLICY):
        print('delete',NEW_DEVICE_POLICY)
        os.remove(NEW_DEVICE_POLICY)

    write_to_file(NEW_DEVICE_POLICY,'''<?xml version="1.0" encoding="UTF-8"?>\n\n''')
    write_to_file(NEW_DEVICE_POLICY,'''<policy>\n''')

def write_to_new_xml_content(items, items_packages):
    global NEW_DEVICE_POLICY
    i = 0
    for item in items:
        if i == 0 or i == len(items)-1:
            write_to_file(NEW_DEVICE_POLICY,' '*4)
            write_to_file(NEW_DEVICE_POLICY,item)
            write_to_file(NEW_DEVICE_POLICY,'\n')
            i+=1
            continue

        package = items_packages[i]
        if cell_valid(package):
            result = cross_xml(package)
            if result == False:
                write_to_file(NEW_DEVICE_POLICY,' '*8)
                write_to_file(NEW_DEVICE_POLICY,item)
                write_to_file(NEW_DEVICE_POLICY,'\n')

        i+=1

def write_to_new_xml_end():
    global NEW_DEVICE_POLICY
    write_to_file(NEW_DEVICE_POLICY,'''</policy>''')

def process_pacakges():
    items_all = {'1':[],'2':[],'8':[],'11':[],'12':[],'13':[],'14':[],'27':[],'28':[],'29':[],'30':[],'31':[]}
    fill_items(items_all)
    write_to_new_xml_header()
    for i in items_all:
        item = items_all[i]
        if len(item) > 0:
            item_xml,items_packages = create_xml_item(item)
            write_to_new_xml_content(item_xml,items_packages)
    write_to_new_xml_end()

def create_xml_item(items_excel):
    global CLASS_INDEX
    global ID_INDEX
    global MEMC_INDEX
    global PACKAGE_INDEX
    items_xml = []
    items_packages = []
    add_header = False
    for i in items_excel:
        package_name = i[PACKAGE_INDEX]
        memc = i[MEMC_INDEX]
        mode_name = i[CLASS_INDEX]
        mode_id = i[ID_INDEX]
        if add_header==False:
            add_header = True
            item  = '''<mode name="%s" id="%s">'''%(mode_name,str(int(mode_id)))
            items_xml.append(item)
            items_packages.append(' ')
        if memc_valid(memc):
            item = '''<item name="package_name" memc="%s">%s</item>'''%(str(int(memc)) ,package_name)
            items_xml.append(item)
            items_packages.append(package_name)
        else:
            item = '''<item name="package_name">%s</item>'''%(package_name)
            items_xml.append(item)
            items_packages.append(package_name)
    end = '''</mode>'''
    items_packages.append(' ')
    items_xml.append(end)
    return items_xml, items_packages

def fill_items(items_all):
    global ID_INDEX
    global PACKAGE_INDEX
    data = xlrd.open_workbook(u'add_more_packages.xlsx')
    table = data.sheet_by_name(u'pacakges')
    nrows = table.nrows
    for i in range(nrows):
        if i == 0:
            #skip the title
            continue

        item = table.row_values(i)
        if cell_valid(item[ID_INDEX]) and cell_valid(item[PACKAGE_INDEX]):
            item_list = items_all[str(int(item[ID_INDEX]))]
            item_list.append(item)

def process_xls():
    process_pacakges()

if __name__ == "__main__":
    if sys.version_info.major != 3:
        print("need python 3.* but get " + str(sys.version_info))
    else :
        process_xls()
