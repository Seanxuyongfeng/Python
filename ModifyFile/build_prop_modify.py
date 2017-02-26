#!/usr/bin/python

import sys
import time

PREFIX_PROP = 'ro.build.id'
def printMsg(msg):
    #sys.stdout.write(msg)
    print(msg)
    sys.stdout.flush()

def removeBackSpace(msg):
    msg = ''.join(msg).strip('\n')
    msg = ''.join(msg).strip('\r\n')
    return msg

def modify_prop(prop_out_path, original_build_id):
    new_build_id_prefix=read_new_prefix(prop_out_path)
    new_build_id_prefix=removeBackSpace(new_build_id_prefix)
    original_build_id = removeBackSpace(original_build_id)
    end_of_origin = original_build_id[-7:]
    printMsg('prefix: ' + new_build_id_prefix)
    printMsg('end 7: ' + end_of_origin)
    new_build_id=''
    if new_build_id_prefix == '':
        new_build_id = original_build_id
    else:
        new_build_id = "".join([new_build_id_prefix,end_of_origin])

    printMsg("new build id is " + new_build_id)
    lines = open(prop_out_path).readlines()
    w_file = open(prop_out_path, 'w')
    for line in lines:
        printMsg(line)
        if line.startswith(PREFIX_PROP+'='):
            w_file.write(line.replace(new_build_id_prefix, "".join([new_build_id_prefix,end_of_origin])))
        else:
            w_file.write(line.replace(original_build_id, new_build_id))
    w_file.close()

def read_new_prefix(prop_out_path):
    global PREFIX_PROP
    fo = open(prop_out_path);
    lines = fo.readlines()
    fo.close();
    for line in lines:
        if line.startswith(PREFIX_PROP+'='):
            new_line = ''.join(line).strip('\n')
            startlen = len(PREFIX_PROP) + 1
            line_len = len(new_line)
            return line[startlen:line_len]
        else:
            continue
    return ''

def main(argv=None):
    if argv is None:
        argv = sys.argv

    modify_prop(argv[1], argv[2])

if __name__ == "__main__":
    sys.exit(main());
