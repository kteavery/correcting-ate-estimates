import matplotlib.pyplot as plt
import csv 
import os
import numpy as np

def plot_results(dirname_worst, dirname_rand):
    avg_frac = np.array([])
    avg_diff = np.array([])
    minimum = 1e10

    dirname = [dirname_worst, dirname_rand]
    xmean = []
    ymean = [] 
    cis = []

    for j in range(0,2):
        for i in range(0, 300):
            f = open(dirname[j]+"/"+str(i)+".csv")
            reader = csv.reader(f)

            if i==0:
                avg_frac = np.array([next(reader)]).astype(float)
                avg_diff = np.array([next(reader)]).astype(float)
                minimum = len(avg_frac[0])
            else:
                new_avg_frac = np.array([next(reader)]).astype(float)
                new_avg_diff = np.array([next(reader)]).astype(float)
                
                minimum = min(minimum, len(new_avg_frac[0]))

                avg_frac = np.append(avg_frac[:,:minimum], new_avg_frac[:,:minimum], axis=0)
                avg_diff = np.append(avg_diff[:,:minimum], new_avg_diff[:,:minimum], axis=0)

        cis.append(1.96 * np.std(np.array(avg_diff).astype(float))/np.sqrt(len(np.array(avg_frac).astype(float))))

        xmean.append(np.mean(avg_frac, axis=0))
        ymean.append(np.mean(avg_diff, axis=0))
    
    fig, ax = plt.subplots()
    ax.plot( xmean[0], ymean[0], color='red' )
    ax.fill_between(xmean[0], (ymean[0]-cis[0]), (ymean[0]+cis[0]), color='red', alpha=.1)

    ax.plot( xmean[1], ymean[1], color="blue", linestyle="dashed")
    ax.fill_between(xmean[1], (ymean[1]-cis[1]), (ymean[1]+cis[1]), color='blue', alpha=.1)

    ax.yaxis.grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', which='both', color='lightgrey', alpha=0.5)

    plt.locator_params(axis='x', nbins=4)
    plt.xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],fontsize = 20)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize = 20)
    plt.xticks(rotation = 45)
    plt.show()


def boxplots(dirname_worst, dirname_rand):
    avg_frac = np.array([])
    avg_diff = np.array([])
    minimum = 1e10

    dirname = [dirname_worst, dirname_rand]
    all_data = []

    for j in range(0,2):
        for i in range(0, 10):
            f = open(dirname[j]+"/"+str(i)+".csv")
            reader = csv.reader(f)

            if i==0:
                frac = np.array([next(reader)]).astype(float)
                data = np.array([next(reader)]).astype(float)
            else:
                new_frac = np.array([next(reader)]).astype(float)
                new_data = np.array([next(reader)]).astype(float)

                data = np.append(data, new_data, axis=0)

        all_data.append(data)
        
    fig, ax = plt.subplots()
    pos = [[1,2,3,4,5,6,7,8,9,10], [1.25,2.25,3.25,4.25,5.25,6.25,7.25,8.25,9.25,10.25]]
    bp1 = ax.boxplot(all_data[0], sym='k+', positions=pos[0])
    bp2 = ax.boxplot(all_data[1], sym='k+', positions=pos[1])
        
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)

    ax.set_ylim(-0.1, 0.3)
    plt.xticks(ticks=[2,4,6,8,10], labels=[0.0005,0.0010,0.0015,0.0020,0.0025])
    
    plt.setp(bp1['whiskers'], color='red', linestyle='-', linewidth=2.0)
    plt.setp(bp1['fliers'], marker='.', markerfacecolor='red', markeredgecolor='red', markersize=6.0)
    plt.setp(bp1['medians'], color='red', linewidth=2.0)
    plt.setp(bp1['caps'], color='red', linewidth=2.0)
    plt.setp(bp1['boxes'], color='red', linewidth=2.0)

    plt.setp(bp2['whiskers'], color='blue', linestyle='-', linewidth=2.0)
    plt.setp(bp2['fliers'], marker='.', markerfacecolor='blue', markeredgecolor='blue', markersize=6.0)
    plt.setp(bp2['medians'], color='blue', linewidth=2.0)
    plt.setp(bp2['caps'], color='blue', linewidth=2.0)
    plt.setp(bp2['boxes'], color='blue', linewidth=2.0)

    plt.show()


if __name__=='__main__':
    # boxplots("/Users/kavery/workspace/correcting-ate-estimates/results/facebook/facebook_greedy_025_0",
    # "/Users/kavery/workspace/correcting-ate-estimates/results/facebook/facebook_random_025_0")

    plot_results("/Users/kavery/workspace/correcting-ate-estimates/results/small_greedy_025_1",
    "/Users/kavery/workspace/correcting-ate-estimates/results/small_random_025_1")