import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
from collections import deque
import pdb
sns.set(style='ticks')

name='TMaze'

seeds = [0,1,2,3,4,]
shortest=np.inf
data=[]
axes=[]
direc='res'
for seed in seeds:
	dat = np.genfromtxt('{}/{}seed{}_intfc1_2opts.csv'.format(direc,name,seed), delimiter=',')[1:200,1]
	print(len(dat))
	if len(dat) < shortest:
		shortest=len(dat)

	rewbuffer = deque(maxlen=100)
	real_dat=[]
	for d in dat:
		rewbuffer.append(d)
		real_dat.append(np.mean(rewbuffer))
	data.append(real_dat)
for i in range(len(data)):
	data[i] = data[i][:shortest]
axes.append(sns.tsplot(data=data,legend=True,condition='IOC',color='red'))



shortest=np.inf
data=[]
for seed in seeds:
	dat = np.genfromtxt('{}/{}seed{}_intfc0_2opts.csv'.format(direc,name,seed), delimiter=',')[1:200,1]
	print(len(dat))
	if len(dat) < shortest:
		shortest=len(dat)

	rewbuffer = deque(maxlen=100)
	real_dat=[]
	for d in dat:
		rewbuffer.append(d)
		real_dat.append(np.mean(rewbuffer))
	data.append(real_dat)
for i in range(len(data)):
	data[i] = data[i][:shortest]
axes.append(sns.tsplot(data=data,legend=True,condition='OC',color='blue'))


plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('Iterations',fontsize=18)
plt.ylabel('Average Rewards',fontsize=18)
plt.legend()
plt.title("Results on {}-v0".format(name))
plt.savefig('plots/{}_notrans.png'.format(name))
plt.clf()
