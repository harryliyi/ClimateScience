import numpy as np


def get_quants(num):
    arr = np.arange(num) + 1
    res = 100*(1-10**(-2*(arr-0.5)/(num-0.5)))
    res_bot = 100*(1-10**(-2*(arr-0.5-0.5)/(num-0.5)))
    res_top = 100*(1-10**(-2*(arr+0.5-0.5)/(num-0.5)))

    return res_top, res, res_bot


num = 4
res_top, res, res_bot = get_quants(num)
print('************************************************')
print(num)
print(res_top)
print(res)
print(res_bot)
