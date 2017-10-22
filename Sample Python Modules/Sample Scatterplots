#This code is part of practicing of the https://github.com/matplotlib/AnatomyOfMatplotlib/blob/master/AnatomyOfMatplotlib-Part2-#Plotting_Methods_Overview.ipynb
import numpy as np
import matplotlib.pyplot as plt

# Our data...
x = np.linspace(0, 10, 100)
y1, y2, y3 = np.cos(x), np.cos(x + 1), np.cos(x + 2)
names = ['Signal 1', 'Signal 2', 'Signal 3']


fig,axes=plt.subplots(nrows=3)

axes[0].set(xlim=(0,10),title=names[0])
#axes[0].plot(x,y1)
axes[0].scatter(x,y1)
axes[1].set(xlim=(0,10),title=names[1])
axes[1].plot(x,y2)
axes[2].set(xlim=(0,10),title=names[2])
axes[2].plot(x,y3)
for ax in axes:
    ax.set(xticks=[],yticks=[])
