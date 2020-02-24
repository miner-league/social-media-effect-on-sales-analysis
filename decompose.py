from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import read_excel

def decompose_sales():
    series = read_excel('data/sales.xlsx', header=0, index_col=0, usecols=[0, 2])
    result = seasonal_decompose(series, model='additive', freq=365)
    result.plot()
    pyplot.show()

if __name__ == '__main__':
    decompose_sales()