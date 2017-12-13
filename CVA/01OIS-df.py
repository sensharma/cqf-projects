import numpy as np
import pandas as pd

print("Expected to take a few seconds to run...")

# Relevant spot UK OIS discount factors for CDS bootstrapping
df_uk_ois = pd.read_excel('in-boedata.xlsx', sheetname='ukois', header=1, index_col=0) 
df_uk_ois.dropna(inplace=True)
df_uk_ois.rename(columns=lambda x: float(str(x)[:4]), inplace=True) 
df_uk_ois.rename(columns={0.08: 0.0}, inplace=True)
df_uk_ois = df_uk_ois/100
uk_ois = df_uk_ois.loc['20151231', :] 
uk_ois = uk_ois[np.arange(0, 5.1, 0.5)]  
df_ois_cds = pd.DataFrame(uk_ois) 
df_ois_cds.to_excel('out10-ois_rates.xlsx')