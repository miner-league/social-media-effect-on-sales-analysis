import re

def remove_commas(string):
    try:
        return re.sub(',', '', string)
    except TypeError as err:
        return string


def remove_percent(string):
    try:
        return re.sub('%', '', string)
    except TypeError as err:
        return string
