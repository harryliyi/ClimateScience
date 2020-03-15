import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

a = [i+1 for i in range(10)]
b = [pow(10, i) for i in range(10)]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(a, b, color='blue', lw=2)

ax.set_yscale('log')


plt.show()
