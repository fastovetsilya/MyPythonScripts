# 1) Import the random function and generate both a random number between 0 and 1 as well as a random number between 1 and 10.
import random as rn
print(rn.random())
print(rn.uniform(1, 10))

# 2) Use the datetime library together with the random number to generate a random, unique value.
from datetime import datetime

year = round(rn.uniform(2000, 2100))
month = round(rn.uniform(1, 12))
day = round(rn.uniform(1, 28))
hour = round(rn.uniform(0, 24))
minute = round(rn.uniform(0, 59))
second = round(rn.uniform(1, 59))

date = datetime(year, month, day, hour, minute, second)
print(date)
