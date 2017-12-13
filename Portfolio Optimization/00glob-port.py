import pandas as pd
from matplotlib import pyplot as plt

from plotsetup import *
plt.style.use('seaborn-muted')

dfNLE = pd.read_excel('04exc_norm-levels.xlsx')
dfER = pd.read_excel('01returns.xlsx')

modify_image()
# plots data and saves an image of the plot in the output folder
fig = plt.figure()
ax = dfNLE.plot()
ax.set_title('Normalized Asset Growth')
ax.set_ylabel('Normalized Value')
ax.set_xlabel('Years')
ax.legend(loc='upper center', ncol=4)
format_axes(plt.axes())
plt.savefig('outB01asset-growth.pdf')

