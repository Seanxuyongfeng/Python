#!/usr/bin/env python

import smtplib

USERNAME = 'xuyongfeng'

def send_email(project,ret):
    smtp = smtplib.SMTP()
    smtp.connect("192.168.0.126", "25")   
    smtp.login('xuyongfeng@droi.com', 'jake9602')   
    smtp.sendmail('xuyongfeng@droi.com', USERNAME+'@droi.com', 
    'From: '+'Compiler Server'+'\r\nTo: '+USERNAME+'@droi.com'+'\r\nSubject: Compile '+ret+'\r\n\r\n'+project)  
    smtp.quit()   

if __name__=="__main__":
	send_email('Project_K506', 'Success');
