import numpy as np
import pyross
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

MM = np.array((0, 0, .2, .2, .2, .2, .2, .2, .4, .4, 1.3, 1.3, 3.6, 3.6, 8, 8))  # mortality per 100

# population and age classes
M = 16  # number of age classes

my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/India_agepop2019.csv')
cols = my_data.keys()
aF = my_data.loc[:, cols[2]]
aM = my_data.loc[:, cols[1]]
Ni = aM + aF
Ni = np.array(Ni[0:M])
Ni = Ni.astype('double')
N = np.sum(Ni)

my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/IN_home.csv')
CH = np.array(my_data)

my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/IN_work.csv')
CW = np.array(my_data)

my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/IN_school.csv')
CS = np.array(my_data)

my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/IN_other.csv')
CO = np.array(my_data)

my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/IN_all.csv')
CA = np.array(my_data)

C = CH + CW + CS + CO

beta = 0.01646692  # contact rate parameter
gIa = 1. / 7  # recovery rate of asymptomatic infectives
gIs = 1. / 7  # recovery rate of symptomatic infectives
alpha = 0.  # asymptomatic fraction
fsa = 1.  # suppresion of contact by symptomatics

# initial conditions
Is_0 = np.zeros(M)
Is_0[6:13] = 3
Is_0[2:6] = 1
Ia_0 = np.zeros(M)
R_0 = np.zeros(M)
S_0 = (Ni - (Ia_0 + Is_0 + R_0))

parameters = {'alpha': alpha, 'beta': beta, 'gIa': gIa, 'gIs': gIs, 'fsa': fsa}
model = pyross.deterministic.SIR(parameters, M, Ni)


# the contact matrix is time-dependent]
# One Lockdown
def contactMatrix(t):
    if t < 12:
        xx = C
    elif 12 <= t < 33:
        xx = CO + CH + CW
    elif 33 <= t < 52:
        xx = CO + CH
    else:
        xx = C
    return xx


# start simulation
Tf = 105
Nf = 2000
data = model.simulate(S_0, Ia_0, Is_0, contactMatrix, Tf, Nf)

IC = np.zeros(2000)
for i in range(M):
    IC += data['X'][:, 2 * M + i]
t = data['t']
fig = plt.figure(num=None, figsize=(28, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 26})

plt.plot(t, IC, '-', lw=4, color='#A60628', label='forecast', alpha=0.8)
plt.xticks(np.arange(0, 200, 7), (
    '14 Mar', '21 Mar', '28 Mar', '4 Apr', '11 Apr', '18 Apr', '25 Apr', '2 May', '9 May', '16 May', '23 May', '30 May',
    '6 Jun', '13 Jun', '20 Jun'), rotation=60)
t1 = int(Nf / 15)
plt.fill_between(t[1 * t1 + 95:7 * t1 + 59], 0, 6000, color='blue', alpha=0.2)
my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/india_tt.csv')
cols = my_data.keys()
day = my_data.loc[:, cols[0]]
cases = my_data.loc[:, cols[1]]
plt.plot(cases, 'o-', lw=4, color='#348ABD', ms=16, label='data', alpha=0.5)
plt.legend(fontsize=26, loc='upper left')
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel('Infected individuals')
plt.ylim(0, 6000)
plt.xlim(0, 105)
plt.show()


# Two Lockdowns
def contactMatrix(t):
    if t < 12:
        xx = C
    elif 12 <= t < 33:
        xx = CO + CH + CW
    elif 33 <= t < 52:
        xx = CO + CH
    elif 52 <= t < 58:
        xx = C - CS
    elif 58 <= t < 82:
        xx = CH
    else:
        xx = C
    return xx


# start simulation
Tf = 105
Nf = 2000
data = model.simulate(S_0, Ia_0, Is_0, contactMatrix, Tf, Nf)

IC = np.zeros(Nf)
SC = np.zeros(Nf)

for i in range(M):
    IC += data['X'][:, 2 * M + i]

fig = plt.figure(num=None, figsize=(28, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 26})

plt.plot(t, IC, '-', lw=4, color='#A60628', label='forecast', alpha=0.8)
plt.xticks(np.arange(0, 200, 7), (
    '14 Mar', '21 Mar', '28 Mar', '4 Apr', '11 Apr', '18 Apr', '25 Apr', '2 May', '9 May', '16 May', '23 May', '30 May',
    '6 Jun', '13 Jun', '20 Jun'), rotation=60)
t1 = int(Nf / 15)
plt.fill_between(t[1 * t1 + 95:7 * t1 + 59], 0, 6000, color='blue', alpha=0.2)
plt.fill_between(t[8 * t1 + 36:11 * t1 + 99], 0, 6000, color='red', alpha=0.2)

my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/india_tt.csv')
cols = my_data.keys()
day = my_data.loc[:, cols[0]]
cases = my_data.loc[:, cols[1]]
plt.plot(cases, 'o-', lw=4, color='#348ABD', ms=16, label='data', alpha=0.5)
plt.legend(fontsize=26)
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel('Infected individuals')
plt.xlim(0, 105)
plt.ylim(0, 6000)
plt.show()


# Three Lockdowns
def contactMatrix(t):
    if t < 12:
        xx = C
    elif 12 <= t < 33:
        xx = CO + CH + CW
    elif 33 <= t < 52:
        xx = CO + CH
    elif 52 <= t < 58:
        xx = C - CS
    elif 58 <= t < 80:
        xx = CH
    elif 80 <= t < 87:
        xx = C - CS
    elif 87 <= t < 105:
        xx = CH
    else:
        xx = C
    return xx


# start simulation
Tf = 105
Nf = 2000
data = model.simulate(S_0, Ia_0, Is_0, contactMatrix, Tf, Nf)

IC = np.zeros(Nf)
SC = np.zeros(Nf)

for i in range(M):
    IC += data['X'][:, 2 * M + i]

fig = plt.figure(num=None, figsize=(28, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 26})

plt.plot(t, IC, '-', lw=4, color='#A60628', label='forecast', alpha=0.8)
plt.xticks(np.arange(0, 200, 7), (
    '14 Mar', '21 Mar', '28 Mar', '4 Apr', '11 Apr', '18 Apr', '25 Apr', '2 May', '9 May', '16 May', '23 May', '30 May',
    '6 Jun', '13 Jun', '20 Jun'), rotation=60)

t1 = int(Nf / 15)
plt.fill_between(t[1 * t1 + 95:7 * t1 + 59], 0, 6000, color='blue', alpha=0.2)
plt.fill_between(t[8 * t1 + 36:11 * t1 + 61], 0, 6000, color='red', alpha=0.2)
plt.fill_between(t[12 * t1 + 60:15 * t1 + 4], 0, 6000, color='red', alpha=0.2)

my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/india_tt.csv')
cols = my_data.keys()
day = my_data.loc[:, cols[0]]
cases = my_data.loc[:, cols[1]]
plt.plot(cases, 'o-', lw=4, color='#348ABD', ms=16, label='data', alpha=0.5)
plt.legend(fontsize=26)
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel('Infected individuals')
plt.xlim(0, 105)
plt.ylim(0, 6000)
plt.show()


# Single Long Lockdown
def contactMatrix(t):
    if t < 12:
        xx = C
    elif 12 <= t < 33:
        xx = CO + CH + CW
    elif 33 <= t < 52:
        xx = CO + CH
    elif 52 <= t < 105:
        xx = CH
    else:
        xx = C
    return xx


# start simulation
Tf = 105
Nf = 2000
data = model.simulate(S_0, Ia_0, Is_0, contactMatrix, Tf, Nf)

IC = np.zeros(Nf)

for i in range(M):
    IC += data['X'][:, 2 * M + i]

t = data['t']
t1 = int(Nf / 15)
fig = plt.figure(num=None, figsize=(28, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 26})

plt.plot(t[0:15 * t1 + 5], IC[0:15 * t1 + 5], '-', lw=4, color='#A60628', label='forecast', alpha=0.6)

plt.xticks(np.arange(0, 200, 7), (
    '14 Mar', '21 Mar', '28 Mar', '4 Apr', '11 Apr', '18 Apr', '25 Apr', '2 May', '9 May', '16 May', '23 May', '30 May',
    '6 Jun', '13 Jun', '20 Jun'), rotation=60)
plt.fill_between(t[1 * t1 + 95:7 * t1 + 59], 0, 6000, color='blue', alpha=0.2)
plt.fill_between(t[7 * t1 + 60:15 * t1 + 5], 0, 6000, color='red', alpha=0.2)

my_data = pd.read_csv('/mnt/c/users/kiran/PycharmProjects/COVID-19/india_tt.csv')
cols = my_data.keys()
day = my_data.loc[:, cols[0]]
cases = my_data.loc[:, cols[1]]
plt.plot(cases, 'o-', lw=4, color='#348ABD', ms=16, label='data', alpha=0.5)
plt.legend(fontsize=26, loc='upper left')
plt.grid()

plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel('Infected individuals')
plt.ylim(0, 6000)
plt.xlim(0, 105)
plt.show()
