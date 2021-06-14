import os
from sys import exit
from datetime import datetime, timedelta
from uuid import uuid4 as uuid
from subprocess import check_output, call, STDOUT

import argparse
import json
import pandas as pd
import re

f = open("results_train_test.txt", 'r')
lines = f.readlines()
f.close()

print("#,Macros,edges,GoldWL,Oneshot,Best,BestEpisode")
for line in lines:
    if len(line.split())<1:
      continue
    if len(line.split())==1:
      print(line.strip(), sep='', end='')
      continue
    if line.startswith('numberOfMacros'):
      values = line.split(":")
      #print(values)
      print(",", values[1].split()[0].strip(), sep='', end='')
      print(",", values[2].split()[0].strip(), sep='', end='')
      print(",", values[3].strip(), sep='', end='')
      continue
    if line.startswith('OneShot'):
      values = line.split(":")
      print(",", values[1].split()[0], sep='', end='')
      continue
    if line.startswith('Best'):
      values = line.split(":")
      print(",", values[1].split()[0].strip(), sep='', end='')
      print(",", values[2].strip(), sep='')
      continue
