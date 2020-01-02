import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
from collections import deque
import pdb
sns.set(style='ticks')
sns.set_context("paper")
sns.set_context("paper", font_scale=1.25)
plt.rcParams['savefig.facecolor'] = "0.8"
plt.rcParams['savefig.edgecolor'] = "black"

name='miniworld-oneroom'

seeds = [0,1,2,3,4,5,6,7,8,9]
#seeds = [0,1,2,3,4]
axes=[]


direc='res/lr0.0001/'
# #--------------------------
shortest=np.inf
data=[]
intlr='_intlr0.0003'
for seed in seeds:
	dat = np.genfromtxt('{}/{}seed{}_intfc1{}_2opts.csv'.format(direc,name,seed,intlr), delimiter=',')[1:385,1]
	# pdb.set_trace()
	print(len(dat))
	if len(dat) < shortest:
		shortest=len(dat)

	rewbuffer = deque(maxlen=100)
	real_dat=[]
	# pdb.set_trace()
	for d in dat:
		rewbuffer.append(d)
		real_dat.append(np.mean(rewbuffer))

	# rewards=[]
	# tot=0
	# for i in range(len(dat)-1):
	# 	if dat[i+1] - dat[i] <399:
	# 		tot+=1
	# 	rewards.append(tot)
	# dat=rewards
	data.append(real_dat)
for i in range(len(data)):
	data[i] = data[i][:shortest]
#axes.append(sns.tsplot(data=data,legend=True,condition='IOC with misspecified $\pi_{\Omega}$',color='red'))
axes.append(sns.tsplot(data=data,legend=True,condition='IOC with uniform fixed $\pi_{\Omega}$',color='red'))




# #--------------------------
# shortest=np.inf
# data=[]
# intlr='_intlr8e-05'
# for seed in seeds:
# 	dat = np.genfromtxt('{}/{}seed{}_intfc1{}_2opts.csv'.format(direc,name,seed,intlr), delimiter=',')[1:385,1]
# 	# pdb.set_trace()
# 	print(len(dat))
# 	if len(dat) < shortest:
# 		shortest=len(dat)
#
# 	rewbuffer = deque(maxlen=100)
# 	real_dat=[]
# 	# pdb.set_trace()
# 	for d in dat:
# 		rewbuffer.append(d)
# 		real_dat.append(np.mean(rewbuffer))
#
# 	# rewards=[]
# 	# tot=0
# 	# for i in range(len(dat)-1):
# 	# 	if dat[i+1] - dat[i] <399:
# 	# 		tot+=1
# 	# 	rewards.append(tot)
# 	# dat=rewards
# 	data.append(real_dat)
# for i in range(len(data)):
# 	data[i] = data[i][:shortest]
# axes.append(sns.tsplot(data=data,legend=True,condition="IOC,"+ r'$\alpha_z$={:.0E}'.format(.00008),color='magenta'))


#
# # #--------------------------
# shortest=np.inf
# data=[]
# intlr='_intlr0.003'
# for seed in seeds:
# 	dat = np.genfromtxt('{}/{}seed{}_intfc1{}_4opts.csv'.format(direc,name,seed,intlr), delimiter=',')[1:385,1]
# 	# pdb.set_trace()
# 	print(len(dat))
# 	if len(dat) < shortest:
# 		shortest=len(dat)
#
# 	rewbuffer = deque(maxlen=100)
# 	real_dat=[]
# 	# pdb.set_trace()
# 	for d in dat:
# 		rewbuffer.append(d)
# 		real_dat.append(np.mean(rewbuffer))
#
# 	# rewards=[]
# 	# tot=0
# 	# for i in range(len(dat)-1):
# 	# 	if dat[i+1] - dat[i] <399:
# 	# 		tot+=1
# 	# 	rewards.append(tot)
# 	# dat=rewards
# 	data.append(real_dat)
# for i in range(len(data)):
# 	data[i] = data[i][:shortest]
# axes.append(sns.tsplot(data=data,legend=True,condition='IOC, ILR 3E-3',color='magenta'))

#
#
#
# # #--------------------------
# shortest=np.inf
# data=[]
# intlr='_intlr0.0008'
# for seed in seeds:
# 	dat = np.genfromtxt('{}/{}seed{}_intfc1{}_4opts.csv'.format(direc,name,seed,intlr), delimiter=',')[1:385,1]
# 	# pdb.set_trace()
# 	print(len(dat))
# 	if len(dat) < shortest:
# 		shortest=len(dat)
#
# 	rewbuffer = deque(maxlen=100)
# 	real_dat=[]
# 	# pdb.set_trace()
# 	for d in dat:
# 		rewbuffer.append(d)
# 		real_dat.append(np.mean(rewbuffer))
#
# 	# rewards=[]
# 	# tot=0
# 	# for i in range(len(dat)-1):
# 	# 	if dat[i+1] - dat[i] <399:
# 	# 		tot+=1
# 	# 	rewards.append(tot)
# 	# dat=rewards
# 	data.append(real_dat)
# for i in range(len(data)):
# 	data[i] = data[i][:shortest]
# axes.append(sns.tsplot(data=data,legend=True,condition='IOC, ILR 8E-4',color='green'))
#
#
#
# # #--------------------------
# shortest=np.inf
# data=[]
# intlr='_intlr8e-05'
# for seed in seeds:
# 	dat = np.genfromtxt('{}/{}seed{}_intfc1{}_4opts.csv'.format(direc,name,seed,intlr), delimiter=',')[1:385,1]
# 	# pdb.set_trace()
# 	print(len(dat))
# 	if len(dat) < shortest:
# 		shortest=len(dat)
#
# 	rewbuffer = deque(maxlen=100)
# 	real_dat=[]
# 	# pdb.set_trace()
# 	for d in dat:
# 		rewbuffer.append(d)
# 		real_dat.append(np.mean(rewbuffer))
#
# 	# rewards=[]
# 	# tot=0
# 	# for i in range(len(dat)-1):
# 	# 	if dat[i+1] - dat[i] <399:
# 	# 		tot+=1
# 	# 	rewards.append(tot)
# 	# dat=rewards
# 	data.append(real_dat)
# for i in range(len(data)):
# 	data[i] = data[i][:shortest]
# axes.append(sns.tsplot(data=data,legend=True,condition='IOC, ILR 8E-5',color='yellow'))
#
#
# # #--------------------------
# shortest=np.inf
# data=[]
# intlr='_intlr0.0005'
# for seed in seeds:
# 	dat = np.genfromtxt('{}/{}seed{}_intfc1{}_4opts.csv'.format(direc,name,seed,intlr), delimiter=',')[1:385,1]
# 	# pdb.set_trace()
# 	print(len(dat))
# 	if len(dat) < shortest:
# 		shortest=len(dat)
#
# 	rewbuffer = deque(maxlen=100)
# 	real_dat=[]
# 	# pdb.set_trace()
# 	for d in dat:
# 		rewbuffer.append(d)
# 		real_dat.append(np.mean(rewbuffer))
#
# 	# rewards=[]
# 	# tot=0
# 	# for i in range(len(dat)-1):
# 	# 	if dat[i+1] - dat[i] <399:
# 	# 		tot+=1
# 	# 	rewards.append(tot)
# 	# dat=rewards
# 	data.append(real_dat)
# for i in range(len(data)):
# 	data[i] = data[i][:shortest]
# axes.append(sns.tsplot(data=data,legend=True,condition='IOC, ILR 5E-4',color='pink'))
#
#
#
# # #--------------------------
# shortest=np.inf
# data=[]
# intlr='_intlr0.0003'
# for seed in seeds:
# 	dat = np.genfromtxt('{}/{}seed{}_intfc1{}_4opts.csv'.format(direc,name,seed,intlr), delimiter=',')[1:385,1]
# 	# pdb.set_trace()
# 	print(len(dat))
# 	if len(dat) < shortest:
# 		shortest=len(dat)
#
# 	rewbuffer = deque(maxlen=100)
# 	real_dat=[]
# 	# pdb.set_trace()
# 	for d in dat:
# 		rewbuffer.append(d)
# 		real_dat.append(np.mean(rewbuffer))
#
# 	# rewards=[]
# 	# tot=0
# 	# for i in range(len(dat)-1):
# 	# 	if dat[i+1] - dat[i] <399:
# 	# 		tot+=1
# 	# 	rewards.append(tot)
# 	# dat=rewards
# 	data.append(real_dat)
# for i in range(len(data)):
# 	data[i] = data[i][:shortest]
# axes.append(sns.tsplot(data=data,legend=True,condition='IOC, ILR 3E-4',color='cyan'))
#
#
# # #--------------------------
# shortest=np.inf
# data=[]
# intlr='_intlr9e-05'
# for seed in seeds:
# 	dat = np.genfromtxt('{}/{}seed{}_intfc1{}_4opts.csv'.format(direc,name,seed,intlr), delimiter=',')[1:385,1]
# 	# pdb.set_trace()
# 	print(len(dat))
# 	if len(dat) < shortest:
# 		shortest=len(dat)
#
# 	rewbuffer = deque(maxlen=100)
# 	real_dat=[]
# 	# pdb.set_trace()
# 	for d in dat:
# 		rewbuffer.append(d)
# 		real_dat.append(np.mean(rewbuffer))
#
# 	# rewards=[]
# 	# tot=0
# 	# for i in range(len(dat)-1):
# 	# 	if dat[i+1] - dat[i] <399:
# 	# 		tot+=1
# 	# 	rewards.append(tot)
# 	# dat=rewards
# 	data.append(real_dat)
# for i in range(len(data)):
# 	data[i] = data[i][:shortest]
# axes.append(sns.tsplot(data=data,legend=True,condition='IOC 9E-5',color='black'))


direc='res/lr0.0001/'
#--------------------------
shortest=np.inf
data=[]
for seed in seeds:
	dat = np.genfromtxt('{}/{}seed{}_intfc0_2opts.csv'.format(direc,name,seed), delimiter=',')[1:385,1]
	print(len(dat))
	if len(dat) < shortest:
		shortest=len(dat)

	rewbuffer = deque(maxlen=100)
	real_dat=[]
	# pdb.set_trace()
	for d in dat:
		rewbuffer.append(d)
		real_dat.append(np.mean(rewbuffer))

	# rewards=[]
	# tot=0
	# for i in range(len(dat)-1):
	# 	if dat[i+1] - dat[i] <399:
	# 		tot+=1
	# 	rewards.append(tot)
	# dat=rewards
	data.append(real_dat)
for i in range(len(data)):
	data[i] = data[i][:shortest]
#axes.append(sns.tsplot(data=data,legend=True,condition='OC with misspecified $\pi_{\Omega}$',color='blue'))
axes.append(sns.tsplot(data=data,legend=True,condition='OC with uniform fixed $\pi_{\Omega}$',color='blue'))


#--------------------------


# direc='res/ppo/lr0.0003/'
# #--------------------------
# shortest=np.inf
# data=[]
# for seed in seeds:
# 	dat = np.genfromtxt('{}/{}seed{}_intfc0_1opts.csv'.format(direc,name,seed), delimiter=',')[1:385,1]
# 	print(len(dat))
# 	if len(dat) < shortest:
# 		shortest=len(dat)
#
# 	rewbuffer = deque(maxlen=100)
# 	real_dat=[]
# 	# pdb.set_trace()
# 	for d in dat:
# 		rewbuffer.append(d)
# 		real_dat.append(np.mean(rewbuffer))
#
# 	# rewards=[]
# 	# tot=0
# 	# for i in range(len(dat)-1):
# 	# 	if dat[i+1] - dat[i] <399:
# 	# 		tot+=1
# 	# 	rewards.append(tot)
# 	# dat=rewards
# 	data.append(real_dat)
# for i in range(len(data)):
# 	data[i] = data[i][:shortest]
# axes.append(sns.tsplot(data=data,legend=True,condition='PPO',color='green'))
#--------------------------



#plt.axvline(x=150, color='k', linestyle='--')
#plt.text(150, 0.3, 'Task Changes', ha='center', va='center',rotation='vertical', bbox={'facecolor':'white', 'pad':5})

plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('Iterations',fontsize=16)
plt.ylabel('Average Rewards',fontsize=16)
plt.legend(prop={'size': 12})

plt.xlabel('Iterations')
plt.ylabel('Average Rewards')
plt.title("MiniWorld OneRoom".format(name))
plt.savefig('plots/learnpio/{}_2Options_swept_misspecPiO.pdf'.format(name), dpi=200, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=True, bbox_inches='tight', pad_inches=0,
            frameon=None, figsize=(4, 4))
plt.clf()
