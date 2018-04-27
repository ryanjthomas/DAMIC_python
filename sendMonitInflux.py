#!/bin/python2.7
import sys
import datetime
from influxdb import client as influxdb
from time import gmtime, strftime

import random

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

influxServer = "monitor.silicon.cf"

currTime = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
dbName = sys.argv[1]
measurementName = sys.argv[2]
if(is_number(sys.argv[3]) == False):
  print 'Error: invalid measurement value, should be a number'
  sys.exit(1)
value = float(sys.argv[3])

client = influxdb.InfluxDBClient(influxServer, 8086, "root", "root", dbName)
tag      = ""
tagValue = ""
if len(sys.argv) >= 6:
	tag = sys.argv[4]
	tagValue = sys.argv[5]
if len(sys.argv) >=7:
  currTime=sys.argv[6]
json_body = [
{
"measurement": measurementName,
"tags": {
tag : tagValue
},
"time": currTime,
"fields": {
"value": value
}
}
]
print json_body
client.write_points(json_body)
