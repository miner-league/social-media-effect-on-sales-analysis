import re

def remove_commas(string):
    try:
        return re.sub(',', '', string)
    except TypeError as err:
        return 0.0
