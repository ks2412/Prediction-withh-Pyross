import numpy as np
import pyross
import matplotlib.pyplot as plt
from timeit import default_timer as timer

M = 1  # the SIR model has no age structure
Ni = 1380000000 * np.ones(M)  # so there is only one age group
N = np.sum(Ni)  # and the total population is the size of this age group

beta = 0.2  # infection rate
gIa = 0.1  # recovery rate of asymptomatic infectives
gIs = 0.8  # recovery rate of symptomatic infectives
alpha = 0  # fraction of asymptomatic infectives
fsa = 1  # the self-isolation parameter

Ia0 = np.array([0])  # the SIR model has only one kind of infective
Is0 = np.array([1000])  # we take these to be symptomatic
R0 = np.array([0])  # and assume there are no recovered individuals initially
S0 = N - (Ia0 + Is0 + R0)  # so that the initial susceptibles are obtained from S + Ia + Is + R = N


# there is no contact structure
def contactMatrix(t):
    return np.identity(M)


# duration of simulation and data file
Tf = 50
Nt = 50

# threshold values
# note that these values are just for demonstration, and in practice
# are too low to assume that stochastic effects become irrelevant
Ias = np.array([0])
Iss = np.array([1000000])
Ss = np.array([1000000000])

# if the simulation is in the stochastic mode and all numbers pass
# the following threshold, the simulation switches to deterministic dynamics
thresholds_from_below = (Ss, Ias, Iss)

# if the simulation is in the deterministic mode and all numbers pass
# the following threshold, the simulation switches to stochastic dynamics
Ias2 = np.array([0])
Iss2 = np.array([500])
Ss2 = np.array([1000])
thresholds_from_above = (Ss2, Ias2, Iss2)

thresholds = {'from_below': thresholds_from_below,
              'from_above': thresholds_from_above}

# instantiate model
parameters = {'alpha': alpha, 'beta': beta, 'gIa': gIa, 'gIs': gIs, 'fsa': fsa}
model = pyross.hybrid.SIR(parameters, M, Ni)

# simulate model
data = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, thresholds)

S = data['X'][:, 0].flatten()
Ia = data['X'][:, 1].flatten()
Is = data['X'][:, 2].flatten()
t = data['t']

fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})

plt.fill_between(t, 0, S / N, color="#348ABD", alpha=0.3)
plt.plot(t, S / N, '-', color="#348ABD", label='$S$', lw=4)

plt.fill_between(t, 0, Is / N, color='#A60628', alpha=0.3)
plt.plot(t, Is / N, '-', color='#A60628', label='$I$', lw=4)

R = N - S - Ia - Is
plt.fill_between(t, 0, R / N, color="dimgrey", alpha=0.3)
plt.plot(t, R / N, '-', color="dimgrey", label='$R$', lw=4)

plt.legend(fontsize=26)
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

'''
N_runs = 100

trajectories_h = np.zeros([N_runs, Nt + 1, 3], dtype=float)
start_h = timer()
cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, thresholds)
for i in range(N_runs - 1):
    print("Simulating trajectory {0} of {1}".format(i + 1, N_runs), end='\r')
    cur_result = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, thresholds)
    trajectories_h[i] = cur_result['X']
end_h = timer()
print("{0} simulations finished in {1:3.2f} seconds".format(N_runs, end_h - start_h))
t_h = cur_result['t']
mean_h = np.mean(trajectories_h, axis=0)
std_h = np.std(trajectories_h, axis=0)

# plot mean and standard deviation

S_h = mean_h[:, 0].flatten()
Ia_h = mean_h[:, 1].flatten()
Is_h = mean_h[:, 2].flatten()
dS_h = std_h[:, 0].flatten()
dIa_h = std_h[:, 1].flatten()
dIs_h = std_h[:, 2].flatten()
# for the recovered, we still need to calculate mean and variance
R_h_trajectories = N - trajectories_h[:, :, 0] - trajectories_h[:, :, 1] - trajectories_h[:, :, 2]
R_h = np.mean(R_h_trajectories, axis=0)
dR_h = np.std(R_h_trajectories, axis=0)

fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})

# hybrid
plt.fill_between(t_h, (S_h - dS_h) / N, (S_h + dS_h) / N, color="#348ABD", alpha=0.2)
plt.plot(t_h, S_h / N, '-', color="#348ABD", label=r'$\langle S\rangle$', lw=4)
plt.fill_between(t_h, (Is_h - dIs_h) / N, (Is_h + dIs_h) / N, color="#A60628", alpha=0.2)
plt.plot(t_h, Is_h / N, '-', color="#A60628", label=r'$\langle I\rangle$', lw=4)
plt.fill_between(t_h, (R_h - dR_h) / N, (R_h + dR_h) / N, color="dimgrey", alpha=0.2)
plt.plot(t_h, R_h / N, '-', color="dimgrey", label=r'$\langle R\rangle $', lw=4)

plt.legend(fontsize=26, loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()
'''
