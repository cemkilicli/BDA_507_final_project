from time import time
from datetime import datetime


def strip(exp_date, type):
        stripped_date = datetime.strptime(exp_date, "%Y-%m-%d %H:%M:%S")
        if type == "date":
            return stripped_date.date()
        elif type == "time":
            return stripped_date.time()
        elif type == "year":
            return stripped_date.year
        elif type == "month":
            return stripped_date.month
        elif type == "year":
            return stripped_date.day

def date_subtract(date1,date2):
    stripped_date1 = datetime.strptime(date1, "%Y-%m-%d")
    stripped_date2 = datetime.strptime(date2, "%Y-%m-%d")
    date_difference = stripped_date2 - stripped_date1
    return date_difference

def strip_other(exp_date, type):
    stripped_date = datetime.strptime(exp_date, "%Y-%m-%d")
    if type == "date":
        return stripped_date.date()
    elif type == "time":
        return stripped_date.time()
    elif type == "year":
        return stripped_date.year
    elif type == "month":
        return stripped_date.month
    elif type == "year":
        return stripped_date.day

def adult_per_room(srch_adult_cnt,srch_rm_cnt):
    adult_per_room = srch_adult_cnt / srch_rm_cnt *1.0
    return adult_per_room

