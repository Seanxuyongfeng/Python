#!/usr/bin/env python

import smtplib

USERNAME = 'zhangsan'

def send_email(project,ret):
    smtp = smtplib.SMTP()
    smtp.connect("192.168.0.126", "25")   
    smtp.login('zhangsan@xxx.com', '123456')   
    smtp.sendmail('zhangsan@xxxx.com', USERNAME+'@xxx.com', 
    'From: '+'Compiler Server'+'\r\nTo: '+USERNAME+'@xxx.com'+'\r\nSubject: Compile '+ret+'\r\n\r\n'+project)  
    smtp.quit()   

if __name__=="__main__":
	send_email('Project_XXX', 'Success');
