import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from math import *
from scipy import interpolate
from scipy.integrate import quad as quad
from plotsetup import *


print("Expected to take a few minutes to run ... ")
# N = 1 as given in project brief
# Fixed Rate assumed to be 1%
# tenors increase in steps of 0.5
# Recovery rate = 40%
# No. of simulations being run = 1000 (can be changed)

N=1
fix = 0.01
dtau = 0.5
RR = 0.4
nsims = 1000


def hjm_mc_sim(fwd_proj, dt, sims, vol_data):   
    '''Takes inputs for HJM MC sim and vol data (specific format)'''
    '''Returns HJM simulated forwards array (3D)'''
    start_array = vol_data[-1, 1:].reshape(len(vol_data[-1, 1:]), 1) 
    steps = int(fwd_proj/dt) 
    tau = 0.5
    tenors = vol_data[0,1:].reshape(len(vol_data[0,1:]),1) 
    phi = np.random.randn(steps, 3, sims)  
    fwds = np.zeros((steps+1, len(tenors), sims)) 
    fwds[0, :, :] = start_array  
    for sim in range(sims): 
        for step in range(1, len(fwds)): 
            for t in range(len(tenors)): 
                m_bar = vol_data[1, t+1]*dt 
                v_bar = (((vol_data[2,t+1]*phi[step-1, 0, sim]+
                           vol_data[3,t+1]*phi[step-1, 1, sim]+
                           vol_data[4,t+1]*phi[step-1, 2, sim]))*np.sqrt(dt))
                if t != len(tenors)-1:
                    f_bar = (fwds[step-1, t+1, sim]-fwds[step-1, t, sim])
                else:
                    f_bar = (fwds[step-1, t, sim]-fwds[step-1, t-1, sim]) 
                fwds[step, t, sim] = fwds[step-1, t, sim] + m_bar + v_bar + f_bar/tau*dt 
    return fwds


# Loading the vol data for HJM MC sims prepared earlier
df_vol_data = pd.read_excel('out09-voldata.xlsx')  
mv_arr = df_vol_data.values 


# Calculating libor~ois spreads for implied OIS disc facts at simulation
df_lib_spot = pd.read_excel('in-boedata.xlsx', sheetname='ukblcspotshort', header=1, index_col=0) 
df_ois_spot = pd.read_excel('in-boedata.xlsx', sheetname='ukoisspotshort', header=1, index_col=0) 

df_lib_spot.dropna(inplace=True) 
df_lib_spot = df_lib_spot.loc['20090102':]
df_ois_spot.dropna(inplace=True) 

df_spd = (df_lib_spot.values - df_ois_spot.values)/100
df_spd = df_spd.mean(axis=0)
spd1 = df_spd[:12].mean()
spd2 = df_spd[12:].mean()


# Running the HJM simulation. Nsims = 1000, dt = 0.01 and plotting simulated fwd curves
# This simulates forwards going up to 25 years forward with starting points up to 5y.

print("Starting simulations, please wait...")

df_res = hjm_mc_sim(5, 0.01, nsims, mv_arr) 

l_tenors = [i for i in range(0, 501, 50)]
df_res_tsteps = df_res[l_tenors, :, :] 
df_f01 = df_res_tsteps[:, 0, :]
df_imp_ois = df_f01[:2,:]-spd1
df_imp_ois = np.concatenate((df_imp_ois, (df_f01[2:,:]-spd2)), axis=0)

modify_image(columns=2)
plt.figure()
plt.style.use('seaborn-muted') 
x = np.arange(0, 25.5, 0.5) 
for i in np.arange(0, 11, 2):
    plt.plot(x, df_res_tsteps[i, :, min(1, floor(nsims/2))], label='T = {}Y'.format(i/2))
plt.ylabel('Rate')
plt.xlabel('Tenors')
# plt.ylim(0.02, 0.08)
plt.legend(loc='upper right', ncol=3)
plt.title('Simulated Forward curves (for 1 MC realization)')
plt.savefig('out14-sim-fwd-curves.pdf') 


# Plotting Evolution of forward rates
plt.figure()
plt.plot(df_res[:, 0, 0], label='r_t')
for i in np.arange(10, df_res.shape[1], 10): 
    plt.plot(df_res[:, i, 0], label='fwd_{}y'.format(i/10))      
plt.title('Evolution of Forward rates (for 1 MC realisation) \n at selected tenor points')
plt.ylabel('Rate')
plt.xlabel('Future time-steps (dt=0.01)')
plt.legend(loc='lower center', ncol=3)
plt.savefig('out15-sim-fwd-evol.pdf')  


###### Implied OIS, MTM and Exposure Calculation
ois_DFs = np.zeros(df_imp_ois.shape)
for row in np.arange(11):
    ois_DFs[row,:] = np.exp(-df_imp_ois[row,:] * (row/2+0.5)) 

df_mtm = pd.DataFrame()
df_exp = pd.DataFrame()
for i in range(1, 11):
    df_step = df_res_tsteps[i, 1:(10-i+2), :]
    df_cf = (df_step-fix) * ois_DFs[i:, :] 
    df_cf = df_cf*dtau*N
    arr_mtm = df_cf.sum(axis=0)
    df_mtm[i] = arr_mtm
    df_exp[i] = np.maximum(arr_mtm, 0)
    df_mtm[0] = 0
    df_mtm.sort_index(axis=1, ascending=True, inplace=True)
    df_mtm[10] = 0
    df_exp[0] = 0
    df_exp.sort_index(axis=1, ascending=True, inplace=True)
    df_exp[10] = 0


# Plotting MTM distribution, mean, median and 97.5 percentile
draws = np.random.randint(0, max(1, (nsims-1)), min(30, nsims))    
plt.figure()
x = np.arange(0, 5.1, 0.5)
for i in draws:
    plt.plot(x, df_mtm.iloc[i, :], linestyle='--')  
avg_mtm_plot, = plt.plot(x, df_mtm.mean(axis=0), color='blue', marker = 'o', label='Avg. MTM')
max_mtm_plot, = plt.plot(x, df_mtm.max(axis=0), color='k', linestyle='--', label='Max. MTM')
min_mtm_plot, = plt.plot(x, df_mtm.min(axis=0), color='k', linestyle='--', label='Min. MTM')
plt.xlim(0, 5)
plt.xticks(np.arange(0, 5.1, 0.5))
plt.ylim(-0.15, 0.20)
plt.ylabel('MTM')
plt.xlabel('6m periods (5y)')
plt.legend(handles=[avg_mtm_plot, max_mtm_plot, min_mtm_plot], loc='upper right')
plt.title('Discounted MTM Profile(MC)')
plt.savefig('out16-mc-mtm-profile.pdf') 


# Plotting Total MTM distribution at start (i.e., 'today')
plt.figure()
pd.DataFrame(df_mtm.sum(axis=1)).hist()
plt.xlabel('Sum of Discounted Total MTM at T=0 (Notional(N) = 1)')
plt.title('Sum of Discounted Total MTM (T=0) distribution')
plt.savefig('out17-mtm_dist.pdf') 


# Plotting Exposure distribution, mean, median and 97.5 percentile
plt.figure()
x = np.arange(0, 5.1, 0.5)
for i in draws:
    plt.plot(x, df_exp.iloc[i, :], linestyle='--')  
avg_exp_plot, = plt.plot(x, df_exp.mean(axis=0), color='blue', marker = 'o', label='Avg. Exposure')
max_exp_plot, = plt.plot(x, df_exp.max(axis=0), color='k', linestyle='--', label='Max. MTM')
min_exp_plot, = plt.plot(x, df_exp.min(axis=0), color='k', linestyle='--', label='Min. MTM')
plt.ylim(-0.02, 0.2)
plt.ylabel('Exposure')
plt.xlabel('6m tenors (5y)')
plt.xticks(np.arange(0, 5.1, 0.5))
plt.legend(handles=[avg_exp_plot, max_exp_plot, min_exp_plot], loc='upper right')
plt.title('Discounted Exposure Profile(MC)')
plt.savefig('out18-mc-exp-profile.pdf') 


# Plotting MTM at various tenors
plt.figure()
mtm_sum = df_mtm.mean(axis=0)[1:]
mtm_sum[::-1].cumsum()[::-1].plot(marker='o')
plt.xlabel('6m tenors (5y)')
plt.title('Discounted cumulative MTM values (MC avg.)')
plt.savefig('out19-cum-mtm.pdf') 


# Plotting Exposure Distribution at each tenor, mean and median
plt.figure()
nIQR = 1.5       # n * IQR for the plot
x = np.arange(1, 11)
meanpts = dict(marker='o', markeredgecolor='black', markerfacecolor='red')
data = df_exp[np.arange(1, 11)]
perc = data.quantile(q=0.975, axis=0)
data.boxplot(showmeans=True, meanprops=meanpts, whis=nIQR, return_type = 'dict')
plt.plot(x, perc, linestyle='', marker='s', color='red')
plt.grid() 
mn_line = mlines.Line2D([], [], color='red', linestyle = '', marker='o', label='Means') 
md_line = mlines.Line2D([], [], color='Red', label='Medians')
perc_line = mlines.Line2D([], [], color='Red', linestyle = '', marker = 's', label='97.5 perc')  
plt.legend(handles=[md_line, mn_line, perc_line])
plt.ylim(bottom=-0.02)
plt.xlabel('6m tenor points (5y)')
plt.ylabel('Exposure') 
plt.title('Exposure Distribution (Range: {} IQR)'.format(nIQR))  
plt.savefig('out20-exp-profile.pdf') 


# Loading the bootstrapped survival probs and calculating PDs for CVA calc
df_pd = pd.read_excel('out13-cds_bs.xlsx', sheetname='nobump', parse_cols=[1, 2]) 
addn = pd.DataFrame([[0, 1]], columns=df_pd.columns.values)
df_pd = pd.concat([addn, df_pd], axis=0)
df_pd['pd'] = 1 - df_pd.survprob/df_pd.survprob.shift(1)
df_pd.fillna(0, inplace=True)

# Loading the BUMPED bootstrapped survival probs and calculating PDs for CVA calc
df_pd_bump = pd.read_excel('out13-cds_bs.xlsx', sheetname='bumped', parse_cols=[1, 2])
addn = pd.DataFrame([[0, 1]], columns=df_pd_bump.columns.values)
df_pd_bump = pd.concat([addn, df_pd_bump], axis=0)
df_pd_bump['pd'] = 1 - df_pd_bump.survprob/df_pd.survprob.shift(1)  
df_pd_bump.fillna(0, inplace=True)


# plotting survival probabilities
cds_range = np.arange(0, 5.1, 0.5) 
plt.figure()
plt.plot(cds_range, df_pd.survprob, marker='o')
plt.title('Bootstrapped Survival Probabilities')
plt.xlim(0, 5)
plt.xticks(np.arange(0, 5.1, 0.5))
plt.xlabel('6m periods (5y)')
plt.savefig('out21-surv-prob-bootstrap.pdf')    


# Creating Period approximations for Exposure, Disc Facts and CVA
df_exp_pd = pd.DataFrame()
for i in range(1, 11):
    df_exp_pd[i] = ((df_exp[i-1]+df_exp[i])/2)


ois_df_pd = np.zeros((10, nsims))
for i in range(1, 11):
    ois_df_pd[i-1, :] = ((ois_DFs[i-1, :]+ois_DFs[i, :])/2) 

cva_pd = df_exp_pd.values*np.swapaxes(ois_df_pd, 0, 1)*(df_pd.pd[1:].reshape(1, 10))*(1-RR)
cva_pd = pd.DataFrame(cva_pd) 
cva_pd.columns = np.arange(1, 11)


# Plotting period CVA distribution
plt.figure()
nIQR = 1.5       # n * IQR for the plot
meanpts = dict(marker='o', markeredgecolor='black', markerfacecolor='red')
x = np.arange(1, 11)
data1 = cva_pd[x]  
perc1 = data1.quantile(q=0.975, axis=0) 
data1.boxplot(showmeans=True, meanprops=meanpts, whis=nIQR, return_type = 'dict')           
plt.plot(x, perc1, linestyle='', marker='s', color='red')    
plt.grid() 
mn_line = mlines.Line2D([], [], color='red', linestyle='', marker='o', label='Means')
md_line = mlines.Line2D([], [], color='Red', label='Medians')
perc_line = mlines.Line2D([], [], color='Red', linestyle='', marker='s', label='97.5 perc')
plt.legend(handles=[md_line, mn_line, perc_line])
plt.ylim(bottom=-0.0001)
plt.xlabel('6m periods (5y)')
plt.title('Period CVA Distribution (Range: {} IQR)'.format(nIQR))
plt.savefig('out22-pd_cva.pdf') 


# Creating and plotting distribution for bumped CVA for sensitivity analysis
cva_pd_bump = df_exp_pd.values*np.swapaxes(ois_df_pd, 0, 1)*(df_pd_bump.pd[1:].reshape(1, 10))*(1-RR)
cva_pd_bump = pd.DataFrame(cva_pd_bump) 
cva_pd_bump.columns = np.arange(1, 11)

plt.figure()
nIQR = 1.5       # n * IQR for the plot
meanpts = dict(marker='o', markeredgecolor='black', markerfacecolor='red')
x = np.arange(1, 11)
data2 = cva_pd_bump[x]  
perc2 = data2.quantile(q=0.975, axis=0) 
data2.boxplot(showmeans=True, meanprops=meanpts, whis=nIQR, return_type = 'dict') 
plt.plot(x, perc2, linestyle='', marker='s', color='red')    
plt.grid() 
mn_line = mlines.Line2D([], [], color='red', linestyle = '', marker='o', label='Means') 
md_line = mlines.Line2D([], [], color='Red', label='Medians')
perc_line = mlines.Line2D([], [], color='Red', linestyle = '', marker = 's', label='97.5 perc') 
plt.legend(handles=[md_line, mn_line, perc_line])
plt.ylim(bottom=-0.0001)
plt.xlabel('6m periods (5y)')
plt.title('Bumped Period CVA Distribution \n (Range: {} IQR)'.format(nIQR))
plt.savefig('out23-pd_bump_cva.pdf') 


# Mean based CVA using sum of period CVAs
mean_pd_cva = cva_pd.mean(axis=0)
sum_cva = mean_pd_cva.sum() 

# Median based CVA using sum of period CVAs
median_pd_cva = cva_pd.median(axis=0)
sum_med_cva = median_pd_cva.sum()

# 97.5% percentile based CVA using sum of period CVAs
perc_pd_cva = cva_pd.quantile(q=0.975, axis=0)
sum_perc_cva = perc_pd_cva.sum()

# Mean based BUMPED CVA using sum of period CVAs
mean_pd_cva_bump = cva_pd_bump.mean(axis=0)
sum_cva_bump = mean_pd_cva_bump.sum() 

# Median based BUMPED CVA using sum of period CVAs
median_pd_cva_bump = cva_pd_bump.median(axis=0)
sum_med_cva_bump = median_pd_cva_bump.sum() 

# 97.5% based BUMPED CVA using sum of period CVAs
perc_pd_cva_bump = cva_pd_bump.quantile(q=0.975, axis=0)
sum_perc_cva_bump = perc_pd_cva_bump.sum() 


# CVA calculation using spline interpolation and integration
x = np.arange(1, 11)
y1 = df_exp_pd.mean(axis=0)
y2 = df_pd.pd[1:].reshape(1, 10).flatten(order='C') 
y3 = ois_df_pd.mean(axis=1) 
Exposure = interpolate.UnivariateSpline(x, y1)
PD = interpolate.UnivariateSpline(x, y2)
DF = interpolate.UnivariateSpline(x, y3)
def CVA_integral(x):
    return Exposure(x)*PD(x)*DF(x)
CVA = (1-RR)*quad(CVA_integral, 0, 10)[0]


# Bumped CVA calculation using spline interpolation and integration
x_b = np.arange(1, 11)
y1_b = df_exp_pd.mean(axis=0)
y2_b = df_pd_bump.pd[1:].reshape(1, 10).flatten(order='C') 
y3_b = ois_df_pd.mean(axis=1) 
Exposure_b = interpolate.UnivariateSpline(x_b, y1_b)
PD_b = interpolate.UnivariateSpline(x_b, y2_b)
DF_b = interpolate.UnivariateSpline(x_b, y3_b)
def CVA_integral_b(x):
    return Exposure_b(x)*PD_b(x)*DF_b(x)
CVA_b = (1-RR)*quad(CVA_integral_b, 0, 10)[0]


print('\n')
print('CVA calculation summary:')
print('\n')
print('Period (mid-point) approximation methodology:')
print('CVA based on means={}'.format(sum_cva)) 
print('CVA based on median={}'.format(sum_med_cva)) 
print('CVA based on 97.5 percentile={}'.format(sum_perc_cva)) 
print('Bumped CVA based on means={}'.format(sum_cva_bump)) 
print('Bumped CVA based on median={}'.format(sum_med_cva_bump)) 
print('Bumped CVA based on 97.5 percentile={}'.format(sum_perc_cva_bump)) 
print('\n')
print('Spline Interpolation - Integration methodology:')
print("CVA (mean) from integration after spline interpolation = {}".format(CVA))
print("BUMPED CVA (mean) from integration after spline interpolation = {}".format(CVA_b))
print('\n')