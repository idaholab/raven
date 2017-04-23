"""
BSD-Style CronTab parsing
"""
"""
Code derived from 
http://android-scripting.googlecode.com/hg/python/gdata/samples/oauth/oauth_on_appengine/appengine_utilities/cron.py

Original notice:

Copyright (c) 2008, appengine-utilities project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the appengine-utilities project nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE 
"""

"""
Possible to-do: Add 'next-run' support - what is next time cron will run?
"""

import os
import re
import datetime

class CronTime(object):
    """
    Parse a BSD-style cron timestamp and test if it should be run at a desired time
    See format at http://unixhelp.ed.ac.uk/CGI/man-cgi?crontab+5
    Also supports N/S increment format (e.g. 1/3 for every 3 min starting at min 1), as
        described at http://en.wikipedia.org/wiki/CRON_expression
    
    """

    def __init__(self, cron_string):
        cron = cron_string.split(" ")
        if len(cron) is not 5:
            raise ValueError, 'Invalid cron string. Format: min hour day mon dow'
        self.cron_raw = {
            'min': cron[0],
            'hour': cron[1],
            'day': cron[2],
            'mon': cron[3],
            'dow': cron[4],
        }
        self.cron_compiled = self._validate_cron(self.cron_raw)

    def _validate_cron(self, cron):
        """
        Parse the field to determine whether it is an integer or lists,
        also converting strings to integers where necessary. If passed bad
        values, raises a ValueError.
        """
        parsers = {
            'dow': self._validate_dow,
            'mon': self._validate_mon,
            'day': self._validate_day,
            'hour': self._validate_hour,
            'min': self._validate_min,
        }
        for el in cron:
            parse = parsers[el]
            cron[el] = parse(cron[el])
        return cron
        
    def _get_max(self, t):
        """
        Get maximum number for a given type
        """
        max_num = {'dow': 7,
             'mon': 12,
             'day':31,
             'hour':23,
             'min': 59
             }
          
        return max_num[t]

    def _validate_type(self, v, t):
        """
        Validates that the number (v) passed is in the correct range for the
        type (t). Raise ValueError, if validation fails.

        Valid ranges:
        day of week = 0-7
        month = 1-12
        day = 1-31
        hour = 0-23
        minute = 0-59

        All can * which will then return the range for that entire type.
        """
        if t == "dow":
            if v >= 0 and v <= 7:
                return [v]
            elif v == "*":
                return "*"
            else:
                raise ValueError, "Invalid day of week."
        elif t == "mon":
            if v >= 1 and v <= 12:
                return [v]
            elif v == "*":
                return range(1, 12)
            else:
                raise ValueError, "Invalid month."
        elif t == "day":
            if v >= 1 and v <= 31:
                return [v]
            elif v == "*":
                return range(1, 31)
            else:
                raise ValueError, "Invalid day."
        elif t == "hour":
            if v >= 0 and v <= 23:
                return [v]
            elif v == "*":
                return range(0, 23)
            else:
                raise ValueError, "Invalid hour."
        elif t == "min":
            if v >= 0 and v <= 59:
                return [v]
            elif v == "*":
                return range(0, 59)
            else:
                raise ValueError, "Invalid minute."

    def _validate_list(self, l, t):
        """
        Validates a crontab list. Lists are numerical values seperated
        by a comma with no spaces. Ex: 0,5,10,15

        Arguments:
            l: comma seperated list of numbers
            t: type used for validation, valid values are
                dow, mon, day, hour, min
        """
        elements = l.split(",")
        return_list = []
        # we have a list, validate all of them
        for e in elements:
            if "-" in e:
                return_list.extend(self._validate_range(e, t))
            else:
                try:
                    v = int(e)
                    self._validate_type(v, t)
                    return_list.append(v)
                except:
                    raise ValueError, "Names are not allowed in lists."
        # return a list of integers
        return return_list

    def _validate_range(self, r, t):
        """
        Validates a crontab range. Ranges are 2 numerical values seperated
        by a dash with no spaces. Ex: 0-10

        Arguments:
            r: dash seperated list of 2 numbers
            t: type used for validation, valid values are
                dow, mon, day, hour, min
        """
        elements = r.split('-')
        # a range should be 2 elements
        if len(elements) is not 2:
            raise ValueError, "Invalid range passed: " + str(r)
        # validate the minimum and maximum are valid for the type
        for e in elements:
            self._validate_type(int(e), t)
        # return a list of the numbers in the range.
        # +1 makes sure the end point is included in the return value
        return range(int(elements[0]), int(elements[1]) + 1)

    def _validate_step(self, s, t):
        """
        Validates a crontab step. Steps are complicated. They can
        be based on a range 1-10/2 or just step through all valid
        */2 or 1/2. When parsing times you should always check for step first
        and see if it has a range or not, before checking for ranges because
        this will handle steps of ranges returning the final list. Steps
        of lists is not supported.

        Arguments:
            s: slash seperated string
            t: type used for validation, valid values are
                dow, mon, day, hour, min
        """
        elements = s.split('/')
        # a range should be 2 elements
        if len(elements) is not 2:
            raise ValueError, "Invalid step passed: " + str(s)
        try:
            step = int(elements[1])
        except:
            raise ValueError, "Invalid step provided " + str(s)
        r_list = []
        # if the first element is *, use all valid numbers
        if elements[0] is "*" or elements[0] is "":
            r_list.extend(self._validate_type('*', t))
        # check and see if there is a list of ranges
        elif "," in elements[0]:
            ranges = elements[0].split(",")
            for r in ranges:
                # if it's a range, we need to manage that
                if "-" in r:
                    r_list.extend(self._validate_range(r, t))
                else:
                    try:
                        r_list.extend(int(r))
                    except:
                        raise ValueError, "Invalid step provided " + str(s)
        elif "-" in elements[0]:
            r_list.extend(self._validate_range(elements[0], t))
        else:   #picloud: support single number
            base = int(elements[0]) % step
            if base == 0 and t in ['mon','day']:
                base += step
            r_list.extend(self._validate_range('%d-%d' % (base, self._get_max(t)), t))        
        return range(r_list[0], r_list[-1] + 1, step)

    def _validate_dow(self, dow):
        """
        """
        # if dow is * return it. This is for date parsing where * does not mean
        # every day for crontab entries.
        if dow is "*":
            return dow
        if dow is '?':
            return '*' #convert '?' to '*'
        days = {
        'mon': 1,
        'tue': 2,
        'wed': 3,
        'thu': 4,
        'fri': 5,
        'sat': 6,
        # per man crontab sunday can be 0 or 7.
        'sun': [0, 7],
        }
        if dow in days:
            dow = days[dow]
            return [dow]
        # if dow is * return it. This is for date parsing where * does not mean
        # every day for crontab entries.
        elif dow is "*":
            return dow
        elif "/" in dow:
            return(self._validate_step(dow, "dow"))
        elif "," in dow:
            return(self._validate_list(dow, "dow"))
        elif "-" in dow:
            return(self._validate_range(dow, "dow"))
        else:
            valid_numbers = range(0, 8)
            if not int(dow) in valid_numbers:
                raise ValueError, "Invalid day of week " + str(dow)
            else:
                return [int(dow)]

    def _validate_mon(self, mon):
        months = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12,
        }
        if mon in months:
            mon = months[mon]
            return [mon]
        elif mon is "*" or mon is '?':
            return range(1, 13)
        elif "/" in mon:
            return(self._validate_step(mon, "mon"))
        elif "," in mon:
            return(self._validate_list(mon, "mon"))
        elif "-" in mon:
            return(self._validate_range(mon, "mon"))
        else:
            valid_numbers = range(1, 13)
            if not int(mon) in valid_numbers:
                raise ValueError, "Invalid month " + str(mon)
            else:
                return [int(mon)]

    def _validate_day(self, day):
        if day is "*":
            return range(1, 32)
        elif "/" in day:
            return(self._validate_step(day, "day"))
        elif "," in day:
            return(self._validate_list(day, "day"))
        elif "-" in day:
            return(self._validate_range(day, "day"))
        else:
            valid_numbers = range(1, 31)
            if not int(day) in valid_numbers:
                raise ValueError, "Invalid day " + str(day)
            else:
                return [int(day)]

    def _validate_hour(self, hour):
        if hour is "*":
            return range(0, 24)
        elif "/" in hour:
            return(self._validate_step(hour, "hour"))
        elif "," in hour:
            return(self._validate_list(hour, "hour"))
        elif "-" in hour:
            return(self._validate_range(hour, "hour"))
        else:
            valid_numbers = range(0, 23)
            if not int(hour) in valid_numbers:
                raise ValueError, "Invalid hour " + str(hour)
            else:
                return [int(hour)]

    def _validate_min(self, min):
        if min is "*":
            return range(0, 60)
        elif "/" in min:
            return(self._validate_step(min, "min"))
        elif "," in min:
            return(self._validate_list(min, "min"))
        elif "-" in min:
            return(self._validate_range(min, "min"))
        else:
            valid_numbers = range(0, 59)
            if not int(min) in valid_numbers:
                raise ValueError, "Invalid min " + str(min)
            else:
                return [int(min)]


        
    def should_run(self, test_time):
        """Determine if the cron should run at test_time.
        test_time can be a unix timestamp (an int) or datetime object
        Note that seconds are ignroed
        """
        
        if not isinstance(test_time, datetime.datetime):
            test_time = datetime.datetime.fromtimestamp(test_time)
        test_time = test_time.replace(second=0, microsecond=0)
        
        #check validity:
        if test_time.month not in self.cron_compiled['mon']:
            return False
        if test_time.day not in self.cron_compiled['day']:
            return False
        if test_time.hour not in self.cron_compiled['hour']:
            return False
        if test_time.minute not in self.cron_compiled['min']:
            return False
        #dow is special case as it may be stored as a '*' for everything
        dow = self.cron_compiled['dow']
        if str(dow) == '*':
            return True
        #use isoweekday to match bsd cron standard where sunday is 0
        return test_time.isoweekday() in self.cron_compiled['dow']


"""unit test code"""
if __name__ == '__main__':
    k = CronTime('* * * * *') 
    assert(k.should_run(843848))
    assert(k.should_run(datetime.datetime(year=2010,month=2,day=3,hour=10,minute=0)))
    k = CronTime('1/3 * * * *') 
    assert(k.should_run(datetime.datetime(year=2010,month=2,day=3,hour=10,minute=0)) == False)
    assert(k.should_run(datetime.datetime(year=2010,month=2,day=3,hour=10,minute=31)) == True)
    k = CronTime('1-34/5 */2 1-10 mar *') 
    assert(k.should_run(datetime.datetime(year=2010,month=3,day=3,hour=10,minute=0)) == False)
    assert(k.should_run(datetime.datetime(year=2010,month=3,day=3,hour=10,minute=30)) == False)
    assert(k.should_run(datetime.datetime(year=2010,month=3,day=3,hour=10,minute=36)) == False)
    assert(k.should_run(datetime.datetime(year=2010,month=3,day=3,hour=11,minute=31)) == False)
    assert(k.should_run(datetime.datetime(year=2010,month=3,day=3,hour=10,minute=31)) == True)
    assert(k.should_run(datetime.datetime(year=2012,month=3,day=3,hour=10,minute=31)) == True)
    assert(k.should_run(datetime.datetime(year=2012,month=4,day=3,hour=10,minute=31)) == False)
    
    #day of week
    k = CronTime('1-34/5 */2 * 0/4 wed') 
    assert(k.should_run(datetime.datetime(year=2010,month=8,day=3,hour=10,minute=31)) == False)
    assert(k.should_run(datetime.datetime(year=2010,month=8,day=4,hour=10,minute=31)) == True)
    assert(k.should_run(datetime.datetime(year=2010,month=8,day=5,hour=10,minute=31)) == False)
    
    k = CronTime('1-34/5 */2 * 0/4 3-4') 
    assert(k.should_run(datetime.datetime(year=2010,month=8,day=3,hour=10,minute=31)) == False)
    assert(k.should_run(datetime.datetime(year=2010,month=8,day=4,hour=10,minute=31)) == True)
    assert(k.should_run(datetime.datetime(year=2010,month=8,day=5,hour=10,minute=31)) == True)
    assert(k.should_run(datetime.datetime(year=2010,month=8,day=6,hour=10,minute=31)) == False)
