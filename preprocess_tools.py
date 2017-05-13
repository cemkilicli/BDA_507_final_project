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
        elif type == "day":
            return stripped_date.day

def date_subtract(date1,date2):
    stripped_date1 = datetime.strptime(date1, "%Y-%m-%d")
    stripped_date2 = datetime.strptime(date2, "%Y-%m-%d")
    date_difference = stripped_date2 - stripped_date1
    return date_difference


def create_month_bins(month):
    if month <=3:
        return  1
    elif month > 3 and month <=6:
        return  2
    elif month >6 and month <=9:
        return 3
    elif month >9 and month <=12:
        return 4

def weekend_check(day):
    if day == 6 or day == 5:
        return 1
    else:
        return 0

