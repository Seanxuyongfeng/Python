#!/usr/bin/python

import os
import sys
import time

BUILD_ID_PROPERTY = 'ro.build.id'
BUILD_DISPLAY_ID = 'ro.build.display.id'
BUILD_NUMBER = 'ro.build.version.incremental'
BUILD_FINGERPRINT = 'ro.build.fingerprint'
BUILD_DESC = 'ro.build.description'
PRODUCT_NAME = 'ro.product.name'
PRODUCT_DEVICE = 'ro.product.device'

def printMsg(msg):
    #sys.stdout.write(msg)
    print(msg)
    sys.stdout.flush()

def removeBackSpace(msg):
    msg = ''.join(msg).strip('\n')
    msg = ''.join(msg).strip('\r\n')
    return msg

def replace_build_id(local_prop_path, build_id_server):
    global BUILD_ID_PROPERTY
    prefix = retrieve_prefix(local_prop_path)
    # there is no ro.build.id in local.prop when prefix is empty
    # so donot need replace build id
    if prefix == '':
        printMsg('prefix is null')
        return

    printMsg('prefix: ' + prefix)
    prefix_length = len(prefix)

    build_id = ""
    if prefix_length > 13:
        build_id = prefix
    else:
        suffix = retrieve_suffix(build_id_server, 10)
        printMsg('end 10: ' + suffix)
        build_id = "".join([prefix,suffix])

    printMsg("build id is " + build_id)
    lines = open(local_prop_path).readlines()
    w_file = open(local_prop_path, 'w')
    for line in lines:
        printMsg(line)
        if line.startswith(BUILD_ID_PROPERTY+'='):
            w_file.write(line.replace(prefix, build_id))
        else:
            w_file.write(line.replace(build_id_server, build_id))
    w_file.close()

def retrieve_suffix(build_id_server, count):
    build_id_server = removeBackSpace(build_id_server)
    length = len(build_id_server)
    if length < count:
        return ''
    else:
        return build_id_server[-count:]

def read_prop(prop_file_path, prop_name):
    value = ''
    try:
        fo = open(prop_file_path);
        lines = fo.readlines()
        fo.close()
        for line in lines:
            if line.startswith(prop_name+'='):
                new_line = ''.join(line).strip('\n')
                startlen = len(prop_name) + 1
                line_len = len(new_line)
                value = line[startlen:line_len]
                value = removeBackSpace(value)
                break
            else:
                continue
    except Exception as ex:
        value = ''
        printMsg('file not found ' + prop_file_path)

    return value

def retrieve_prefix(local_prop_path):
    global BUILD_ID_PROPERTY
    prefix = ''
    try:
        fo = open(local_prop_path);
        lines = fo.readlines()
        fo.close()
        for line in lines:
            if line.startswith(BUILD_ID_PROPERTY+'='):
                new_line = ''.join(line).strip('\n')
                startlen = len(BUILD_ID_PROPERTY) + 1
                line_len = len(new_line)
                prefix = line[startlen:line_len]
                prefix = removeBackSpace(prefix)
                break
            else:
                continue
    except Exception as ex:
        prefix = ''
        printMsg('file not found ' + local_prop_path)

    return prefix

def copy_and_rm(output_dir):
    lines = ''
    # read from default.prop
    default_prop_path = output_dir + '/default.prop'
    try:
        r_file = open(default_prop_path)
        lines = r_file.readlines()
        r_file.close()
    except Exception as ex:
        printMsg('read file failed ' + default_prop_path + ' ' + str(ex))

    # write to local.prop
    local_prop_path = output_dir + '/local.prop'
    try:
        if lines != '':
            w_file = open(local_prop_path, 'w+')
            for line in lines:
                w_file.write(removeBackSpace(line)+'\n')
            w_file.close()
    except Exception as ex:
        printMsg('write file failed ' + local_prop_path + ' ' + str(ex))

    try:
        os.remove(default_prop_path)
    except Exception as ex:
        printMsg('remove file failed ' + str(ex))

def empty_line(file_path):
    try:
        a_file = open(file_path, 'a+')
        a_file.write('\n')
        a_file.close()
    except Exception as ex:
        printMsg('empty line failed ' + ' ' + file_path + ' ' + str(ex))

def add_prop(file_path, prop, value):
    try:
        a_file = open(file_path, 'a+')
        a_file.write(prop+'='+value+'\n')
        a_file.close()
    except Exception as ex:
        printMsg('write prop failed ' + prop + ' ' + file_path + ' ' + str(ex))

def build_replaced_fingerprint(local_prop_path, source_fingerprint):
    global PRODUCT_NAME
    global PRODUCT_DEVICE
    out_fingerprint = ''
    productStartIndex = source_fingerprint.find('/')
    if productStartIndex != -1:
        out_fingerprint = source_fingerprint[0:productStartIndex+1]
        product_name = read_prop(local_prop_path, PRODUCT_NAME)
        #add product name
        out_fingerprint = out_fingerprint + product_name + '/'
        deviceNameStartIndex = source_fingerprint.find('/', productStartIndex+1)
        if deviceNameStartIndex != -1:
            device_name = read_prop(local_prop_path, PRODUCT_DEVICE)
            # add device name
            out_fingerprint = out_fingerprint + device_name
            colonIndex = source_fingerprint.find(':', deviceNameStartIndex+1)
            if colonIndex != -1:
                # add rest string from colon
                out_fingerprint = out_fingerprint + source_fingerprint[colonIndex:]
                return out_fingerprint
            else:
                return source_fingerprint
        else:
            return source_fingerprint

        printMsg('out_fingerprint: ' + out_fingerprint)
    else:
        return source_fingerprint

def addition_prop(local_prop_path, build_display_id, build_number, build_fingerprint, build_desc):
    global BUILD_DISPLAY_ID
    global BUILD_NUMBER
    global BUILD_FINGERPRINT
    global BUILD_DESC
    empty_line(local_prop_path)
    add_prop(local_prop_path, BUILD_DISPLAY_ID, build_display_id)
    add_prop(local_prop_path, BUILD_NUMBER, build_number)
    output_fingerprint = build_replaced_fingerprint(local_prop_path, build_fingerprint);
    add_prop(local_prop_path, BUILD_FINGERPRINT, output_fingerprint)
    add_prop(local_prop_path, BUILD_DESC, build_desc)

def main(argv=None):
    if argv is None:
        argv = sys.argv

    output_dir = argv[1]
    copy_and_rm(output_dir)
    local_prop_path = output_dir + '/local.prop'
    build_id_server = argv[2]
    build_display_id = argv[3]
    build_number = argv[4]
    build_fingerprint = argv[5]
    build_desc = argv[6]
    addition_prop(local_prop_path,build_display_id,build_number,build_fingerprint,build_desc)
    replace_build_id(local_prop_path, build_id_server)

if __name__ == "__main__":
    sys.exit(main());