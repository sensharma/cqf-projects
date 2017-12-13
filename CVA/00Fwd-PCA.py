import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.integrate import quad as quad
from plotsetup import *

print("Expected to take a few seconds to run...")

def pca3(cmat):
    '''3 factor PCA on a cov matrix provided'''

    eig_vals, eig_vecs = np.linalg.eigh(cmat)  
    eig_v3 = eig_vals[-3:] 
    eig_v3 = eig_v3[::-1] 
    pc3 = pd.DataFrame(eig_vecs).iloc[:, -3:]
    pc3.rename(columns={50:1, 49:2, 48:3}, inplace=True)
    pc3 = pc3[[1, 2, 3]]
    pc3.iloc[:,:2] = -pc3.iloc[:,:2] 
    eig_v3 = eig_vals[-3:] 
    eig_v3 = eig_v3[::-1] 
    pc3 = pd.DataFrame(eig_vecs).iloc[:, -3:]
    pc3.rename(columns={50:1, 49:2, 48:3}, inplace=True)
    pc3 = pc3[[1, 2, 3]]
    return eig_v3, pc3 


def vol1(x):
    '''linear fitted vol1 - pc1'''
    return coeff1[0] 


def vol2(x):
    '''cubic fitted vol2 - pc2'''
    return (coeff2[0]*x**3 + coeff2[1]*x**2 + coeff2[2]*x + coeff2[3]) 


def vol3(x):
    '''cubic fitted vol2 - pc2'''
    return (coeff3[0]*x**3 + coeff3[1]*x**2 + coeff3[2]*x + coeff3[3])  


def mbar(t):  
    '''returns integrated m-bar for a list of tenors using quadrature'''
    L = [] 
    for t in tau:
        m1 = quad(vol1, 0, t)[0]
        m1 *= vol1(t)
        m2 = quad(vol2, 0, t)[0]
        m2 *= vol2(t)
        m3 = quad(vol3, 0, t)[0]
        m3 *= vol3(t)
        m = m1 + m2 + m3
        L.append(m)
    return L

# loading data and generating the covariance matrix
# data loaded from spreadsheet downloaded from BOE

df = pd.read_excel('in-boedata.xlsx', sheetname='ukblc', header=1, index_col=0) 
df.dropna(axis=0, how='any', inplace=True)
df.rename(columns=lambda x: float(str(x)[:4]), inplace=True) 
df.dropna(axis=0, how='any', inplace=True)
df.rename(columns=lambda x: float(str(x)[:4]), inplace=True) 
df.rename(columns={0.08: 0.0}, inplace=True) 
df_diff = df.diff(periods=1, axis=0).dropna(how='any') 
cmat = df_diff.cov()*252/10000  

# plotting the spot rates (1m fwd as proxy)
modify_image(columns=2)
plt.figure()
plt.style.use('seaborn-muted')
ax = df[0.0].plot()
plt.title('Spot rates')
plt.xlabel('Year')
plt.ylabel('Rate (%)')
plt.savefig('out01-spots.pdf')

# plotting historical forward rates
plt.figure()
for i in [1.0, 5.0, 10.0, 25.0]:
    ax = df[i].plot(label=i)
plt.title('Historical evolution of selected Forward rates')
plt.xlabel('Year')
plt.ylabel('Rate(%)')
plt.legend()
plt.savefig('out02-hist-fwd.pdf')

# plotting evolution of certain historical fwd rates
plt.figure()
for i in [0, 1000, 2000, -1]:
    ax = df.ix[i].plot(label=df.index[i].date())
plt.ylabel('Rate (%)')
plt.xlabel('Tenors')
plt.legend(loc='lower right')
plt.title('Forward curves at various points in time')
plt.savefig('out03-fwd-curve-evol.pdf')

# plotting distribution of historical key fwd rates
plt.figure()
nIQR = 2.5       # n * IQR for the plot
meanpts = dict(marker='o', markeredgecolor='black', markerfacecolor='red')
data = df[[0.0, 1.0, 5.0, 10.0, 20.0, 25.0]] 
ax = data.boxplot(showmeans=True, meanprops=meanpts, whis=nIQR, return_type='dict')
plt.grid() 
mn_line = mlines.Line2D([], [], color='red', linestyle = '', marker='o', label='Means') 
md_line = mlines.Line2D([], [], color='Red', label='Medians')
plt.legend(handles=[md_line, mn_line])
plt.xlabel('Tenors')
plt.ylabel('Rate (%)')
plt.title('Historical key forward rates (Range: {} IQR)'.format(nIQR))
plt.savefig('out04-tboxpl.pdf')


# getting and plotting 3-factor PCA results
eval3, pc3 = pca3(cmat) 

plt.figure()
for count, i in enumerate(pc3.columns):
    ax = pc3.iloc[:, count].plot(label='PC{}'.format(i))
plt.ylim(-0.6, 0.4)
plt.xlim((0, 25))
plt.title('PCA results')
plt.ylabel('EigVector (% move at each tenor)')
plt.xlabel('Tenors')
plt.legend(loc='lower right')
plt.savefig('out05-pca_res.pdf')

# obtaining the vol components corresponding to the PCs
vols = pc3
vols.iloc[:,0] = vols.iloc[:,0] * np.sqrt(eval3[0])
vols.iloc[:,1] = vols.iloc[:,1] * np.sqrt(eval3[1])
vols.iloc[:,2] = vols.iloc[:,2] * np.sqrt(eval3[2]) 

tau = df.columns.values 

# fitting the vol components, the 1st to a constant, the others to cubics
# curve fitting using least squares
coeff1 = np.polyfit(tau, vols.iloc[:, 0], deg=0) 
coeff2 = np.polyfit(tau, vols.iloc[:, 1], deg=3) 
coeff3 = np.polyfit(tau, vols.iloc[:, 2], deg=3)  

# plotting the fitted and original PCs
vf1 = np.ones((1, len(tau)))
vf1 = pd.DataFrame(vf1*coeff1)

plt.figure()
ax = vols.iloc[:, 0].plot()
vf1.ix[0].plot()  
plt.xlim((0, 25))
plt.title('Fitted 1st Principal Component')
plt.xlabel('Tenors')
plt.savefig('out06-v1_fit.pdf')

vf2 = np.ones((1, len(tau)))
vf2 = pd.DataFrame(vf2*coeff2[-1] + coeff2[-2]*tau + coeff2[-3]*tau**2 + coeff2[-4]*tau**3)

plt.figure()
vols.iloc[:,1].plot() 
ax = vf2.ix[0].plot()
plt.xlim((0, 25))
plt.title('Fitted 2nd Principal Component')
plt.xlabel('Tenors')
plt.savefig('out07-v2_fit.pdf')

vf3 = np.ones((1, len(tau)))
vf3 = pd.DataFrame(vf3*coeff3[-1] + coeff3[-2]*tau + coeff3[-3]*tau**2 + coeff3[-4]*tau**3)

plt.figure()
ax = vols.iloc[:, 2].plot()
vf3.ix[0].plot()
plt.xlim((0, 25))
plt.title('Fitted 3rd Principal Component')
plt.xlabel('Tenors')
plt.savefig('out08-v3_fit.pdf')

# preparing and saving the vol data to be used in the HJM MC simulations
df.iloc[-1, :]  
cols = list(np.arange(1, 52))  
cols.insert(0, 'Columns')
df_vol_data = pd.DataFrame(columns=cols) 
df_vol_data.Columns = ['Tenor', 'mbar', 'Vol_1', 'Vol_2', 'Vol_3', 'Start'] 
df_vol_data.iloc[0, 1:] = tau
df_vol_data.iloc[1, 1:] = mbar(tau) 
df_vol_data.iloc[2, 1:] = vf1.ix[0].values
df_vol_data.iloc[3, 1:] = vf2.ix[0].values
df_vol_data.iloc[4, 1:] = vf3.ix[0].values
df_vol_data.iloc[5, 1:] = (df.iloc[-1, :].values)/100
df_vol_data.set_index('Columns', inplace=True) 
df_vol_data.to_excel('out09-voldata.xlsx') 
