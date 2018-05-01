#External Modules--------------------begin
import xml.etree.ElementTree as ET
import os
import sys
import threading
import time
import traceback
import csv
from datetime import datetime, timedelta, date
from collections import defaultdict

#External Modules--------------------end
if __name__ == '__main__':

  def get_outages_for_unit(date, unitname, unit_dict):
    # get unit
    for event in unit_dict[unitname]:
      teor = float(event['Tech cap'].replace("MW", "")) - float(
          event['Avail cap'].replace("MW", ""))
      a = date.year, date.month, date.day
      if a == (event['Start'].year, event['Start'].month, event['Start'].day):
        tomorrow = datetime.now() + timedelta(1)
        midnight = datetime(
            year=tomorrow.year,
            month=tomorrow.month,
            day=tomorrow.day,
            hour=0,
            minute=0,
            second=0)
        weight = 24.0 * 3600 / (midnight - event['Start']).seconds
        return weight * teor
      elif a == (event['End'].year, event['End'].month, event['End'].day):
        yesterday = datetime.now() + timedelta(-1)
        midnight = datetime(
            year=yesterday.year,
            month=yesterday.month,
            day=yesterday.day,
            hour=0,
            minute=0,
            second=0)
        weight = 24.0 * 3600 / (event['End'] - midnight).seconds
        return weight * teor
      else:
        for dd in event['dates_in_between']:
          if a == (dd.year, dd.month, dd.day):
            return teor

  file_obj = open("/Users/alfoa/projects/raven_github/raven/test.csv", "r+")
  reader = csv.DictReader(file_obj, delimiter=',')
  _unit_dict = {}
  for row in reader:
    # get unit name
    unit_name = row.pop('Unit')
    # data
    data = row
    try:
      start = datetime.strptime(row['Start'], "%d/%m/%Y %H:%M")
    except ValueError:
      start = datetime.strptime(row['Start'], "%d/%m/%Y %H:%M:%S")
    try:
      end = datetime.strptime(row['End'], "%d/%m/%Y %H:%M")
    except ValueError:
      end = datetime.strptime(row['End'], "%d/%m/%Y %H:%M:%S")
    data['Start'] = start
    data['End'] = end
    delta = end - start
    data['time_w'] = delta
    data['dates_in_between'] = []
    for i in range(delta.days + 1):
      data['dates_in_between'].append(start + timedelta(days=i))
    # store data
    if unit_name not in _unit_dict:
      _unit_dict[unit_name] = []
    _unit_dict[unit_name].append(data)

  get_outages_for_unit(date(2017, 10, 1), 'TRICASTIN 4', _unit_dict)
