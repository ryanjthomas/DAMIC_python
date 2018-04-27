#!/bin/python

import requests
import sys

api_url='https://air.uchicago.edu/groupmeeting/api.php'

USER='ryant'
PASS='L2eH02EpKsTr'

fname=sys.argv[1]
remote_fname=sys.argv[2]

user_agent="PythonUploader/0.1"

token_payload={'action': 'query', 'format': 'json', 'utf8': '', 
           'meta': 'tokens', 'type': 'login'}

r1=requests.post(api_url, data=token_payload)
login_token=r1.json()['query']['tokens']['logintoken']

login_payload = {'action': 'login', 'format': 'json', 'utf8': '', 
           'lgname': USER, 'lgpassword': PASS, 'lgtoken': login_token}
r2 = requests.post(api_url, data=login_payload, cookies=r1.cookies)
cookies=r2.cookies.copy()

def get_edit_token(cookies):
  edit_token_response=requests.post(api_url, data={'action': 'query',
                                                   'format': 'json', 
                                                   'meta': 'tokens'}, cookies=cookies)
  return edit_token_response.json()['query']['tokens']['csrftoken']

upload_payload={'action': 'upload', 
            'format':'json',
            'filename':remote_fname, 
            'comment':'<MY UPLOAD COMMENT>',
            'text':'Text on the File: page... description, license, etc.',
            'token':get_edit_token(cookies)}

files={'file': (remote_fname, open(fname,'rb'),'pdf',{'Content-type':'pdf'})}

headers={'User-Agent': user_agent}

upload_response=requests.post(api_url, data=upload_payload,files=files,cookies=cookies,headers=headers)

