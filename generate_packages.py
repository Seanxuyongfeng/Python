#!/usr/bin/python
import random
import string
from random import Random

def random_str(randomlength=8):
    str = ''
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz'
    length = len(chars) - 1
    random = Random()
    for i in range(randomlength):
        str+=chars[random.randint(0, length)]
    return str

def write_to_file(msg):
    file = open('packages.txt','a')
    file.write(msg)
    file.close()

def generate_package():
    fisrt = random_str(3)
    second = random_str(5)
    third = random_str(4)
    return fisrt+"."+second+"."+third

def generate_pks_batch(num):
    result = ''
    for i in range(0,num):
        package = generate_package()
        if i < (num-1):
            result = result + package +":"
        else:
            result = result + package
    return result

if __name__ == "__main__":
    packages = generate_pks_batch(10000)
    write_to_file(packages)
