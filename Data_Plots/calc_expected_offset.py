import numpy as np 
from scipy.special import erf

# Area of KV450 (KV) and KiDS-1000 (KT)
A_KV=341.3
A_KT=777.4

#mode with marginal 68% CI errors from Wright et al 2020, vs Asgari et al 2pt
S8_KV=0.716
eS8_KV=(0.043+0.038)/2.0

S8_KT=0.768
eS8_KT=(0.016+0.020)/2.0


#MAP from Wright et al 2020, vs Asgari et al 2pt
S8_KV=0.767
eS8_KV=(0.043+0.038)/2.0

S8_KT=0.768
eS8_KT=(0.022+0.015)/2.0

#mode with marginal 68% CI errors from Troester et al 2020, vs CH/TT et al 2pt
#3x2pt marginal

S8_KV=0.728
eS8_KV=0.026

S8_KT=0.766
eS8_KT=(0.017+0.016)/2.0

diff = S8_KT - S8_KV
#vardiff = eS8_KT**2 + eS8_KV**2 - (2*np.sqrt(A_KV/A_KT)*eS8_KT*eS8_KV)

vardiff= eS8_KT**2 + eS8_KV**2 - 2*(A_KV/A_KT)*eS8_KV**2

print (diff, diff/np.sqrt(vardiff))

#different derivation, same result
#vardiff=eS8_KT**2 +eS8_KV**2*((A_KT-A_KV)**2 - A_KV**2)/A_KT**2
#print (vardiff)

exp_diff= np.sqrt(2*vardiff/np.pi)
cdf= erf(diff/np.sqrt(2*vardiff))

print (exp_diff, exp_diff/np.sqrt(vardiff),cdf)

