#!/usr/bin/env python

import time
from datetime import datetime

# Get the seconds since epoch
secondsSinceEpoch = time.time()

# Convert seconds since epoch to struct_time
timeObj = time.localtime(secondsSinceEpoch)

# get the current timestamp elements from struct_time object i.e.
print('Current TimeStamp is : %d-%d-%d %d:%d:%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec))