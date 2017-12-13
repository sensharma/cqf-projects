import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotsetup import *

plt.style.use('seaborn-muted')
# CDS spread interpolation based on 1y and 5y spreads available
# 1y CDS spd = 78.49bps  5y CDS spd = 134.51bps
# Bumped spreads created for sensitivity analysis: Bump = 10%

cds_range = np.arange(0, 5.1, 0.5) 
scp_cds_spd = pd.DataFrame(np.nan, index=cds_range, columns=['StanC CDS Spd'])
scp_cds_spd.iloc[0, 0] = 0.0
scp_cds_spd.iloc[2, 0] = 0.007849
scp_cds_spd.iloc[10, 0] = 0.013451 
scp_interp = scp_cds_spd.interpolate(method='linear')
scp_interp['Bumped Spd'] = scp_interp['StanC CDS Spd'] + 0.1
scp_interp.iloc[0, 1] = 0
scp_interp.to_excel('out11-cds-spreads.xlsx') 

modify_image(columns=2)
fig = scp_interp.iloc[:,0].plot(x=cds_range, marker='o')
plt.xlim(0, 5)
plt.xticks(np.arange(0, 5.1, 0.5))
fig.set_title('StanChart Interpolated CDS spreads')
fig.set_xlabel('6m periods (5y)')
plt.savefig('out12-cds_spd_interp.pdf')

