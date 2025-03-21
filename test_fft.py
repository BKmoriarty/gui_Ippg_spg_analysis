# Code to add text on matplotlib
 
# Importing library
import matplotlib.pyplot as plt
 
# Creating x-value and y-value of data
x = [1, 2, 3, 4, 5]
y = [5, 8, 4, 7, 5]
 
# Creating figure
fig = plt.figure()
 
# Adding axes on the figure
ax = fig.add_subplot(111)
 
# Plotting data on the axes
ax.plot(x, y)
 
# Adding title
ax.set_title('Day v/s No of Questions on GFG', fontsize=15)
 
# Adding axis title
ax.set_xlabel('Day', fontsize=12)
ax.set_ylabel('No of Questions', fontsize=12)
 
# Setting axis limits
ax.axis([0, 10, 0, 15])
 
# Adding text on the plot.
ax.text(1, 13, 'Practice on GFG', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
 
# Adding text without box on the plot.
ax.text(8, 13, 'December', style='italic')
 
# Adding annotation on the plot.
ax.annotate('Peak', xy=(2, 8), xytext=(4, 10), fontsize=12,
            arrowprops=dict(facecolor='green', shrink=0.05))
 
plt.show()