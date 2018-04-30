#!/bin/python

import requests
import sys
import config

api_url=config.api_url
USER=config.user
PASS=config.password
user_agent=config.user_agent

# fname=sys.argv[1]
# remote_fname=sys.argv[2]

login_token_payload={'action': 'query', 'format': 'json', 'utf8': '', 
           'meta': 'tokens', 'type': 'login'}

login_token_request=requests.post(api_url, data=login_token_payload)
login_token=login_token_request.json()['query']['tokens']['logintoken']

login_payload = {'action': 'login', 'format': 'json', 'utf8': '', 
           'lgname': USER, 'lgpassword': PASS, 'lgtoken': login_token}
login_request = requests.post(api_url, data=login_payload, cookies=login_token_request.cookies)
cookies=login_request.cookies.copy()

def get_edit_token(cookies):
  edit_token_response=requests.post(api_url, data={'action': 'query',
                                                   'format': 'json', 
                                                   'meta': 'tokens'}, cookies=cookies)
  return edit_token_response.json()['query']['tokens']['csrftoken']

# upload_payload={'action': 'upload', 
#             'format':'json',
#             'filename':remote_fname, 
#             'comment':'<MY UPLOAD COMMENT>',
#             'text':'Text on the File: page... description, license, etc.',
#             'token':get_edit_token(cookies)}

# files={'file': (remote_fname, open(fname,'rb'),'pdf',{'Content-type':'pdf'})}

# headers={'User-Agent': user_agent}

# upload_response=requests.post(api_url, data=upload_payload,files=files,cookies=cookies,headers=headers)


#Example code to make edit a page
# requests.post(api_url, data={"action":"edit", "title":"User:Ryant/Sandbox", "section":"new", "sectiontitle":"RunID xxxx", "text":"Test page. \n [[File:Test upload.pdf]]","token":get_edit_token(cookies)}, cookies=cookies)
