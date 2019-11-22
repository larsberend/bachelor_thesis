
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig =plt.figure(figsize=(25, 10))

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=15)
labels = ['Shi-Tomasi',  'Fast Hessian', 'DOG']

# for preprocessing:
# average lifespan
# one_means = [1064.48, 810.66, 938.75]
# two_means = [2650.80, 2869.93, 1328.32]
#
# #average tracked points
# one_tracked = [1004.80, 2200.6, 1473.15]
# two_tracked = [555.00, 863.00, 519]


# # for lk:
# # average lifespan
# one_means = [563.96, 500.32, 336.93]
# two_means = [2129.63, 2605.77, 1328.32]
#
# #average tracked points
# one_tracked = [1733.30, 608.85, 1885.30]
# two_tracked = [562.00, 523.50, 519]

# for detectors
one_means = [307.09, 166.50, 87.69]
two_means = [2105.74, 1166.50, 1226.61]

#average tracked points
one_tracked = [740.45, 874.25, 2941.9]
two_tracked = [573.50, 715.5, 545]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

plt.subplot(121)

rects1 = plt.bar(x - width/2, one_means, width, label='120 Hz',color='steelblue')
rects2 = plt.bar(x + width/2, two_means, width, label='200 Hz',color='lightsteelblue')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Average Lifespan',fontsize=20)
#ax.set_title('Average Lifespan for Keypoint Detectors')
plt.yticks(fontsize=15)
plt.xticks(x, labels,fontsize=15)
plt.legend(fontsize=15)
autolabel(rects1)
autolabel(rects2)

plt.subplot(122)

rects1_tracked = plt.bar(x - width/2, one_tracked, width, label='120 Hz', color='forestgreen')
rects2_tracked = plt.bar(x + width/2, two_tracked, width, label='200 Hz',color='yellowgreen')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Average number of tracked points', fontsize=20)
#ax.set_title('Average Lifespan for Keypoint Detectors')
plt.yticks(fontsize=15)
plt.xticks(x, labels,fontsize=15)
plt.legend(fontsize=15)

autolabel(rects1_tracked)
autolabel(rects2_tracked)
fig.savefig('values_keypointdetectors')

plt.show()
