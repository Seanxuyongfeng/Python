#!/usr/bin/python

import os,shutil
import fnmatch

def all_files(root, patterns='*',single_level=False, yield_folders=False):
    "list dirs and files"
    patterns = patterns.split(';')

    for path, subdirs, files in os.walk(root):
        if yield_folders:
            files.extend(subdirs)
        files.sort()

        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    yield os.path.join(path,name);
                    break;
        if single_level:
            break;
currentDir = os.getcwd()
for path in all_files('D:\dexcompile\ApkTool\QQyinle_806aaa','*'):
    if path.endswith('aidl'):
        currentStr = str(currentDir)
        newPaht = path.replace(currentStr,currentStr+"\out")
        print(path)
        print(newPaht)
        fpath,fname=os.path.split(newPaht)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(path,newPaht)

