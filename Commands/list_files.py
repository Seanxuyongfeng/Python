#!/usr/bin/python

import os
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

for path in all_files('G:\Music','*'):
    print(path)