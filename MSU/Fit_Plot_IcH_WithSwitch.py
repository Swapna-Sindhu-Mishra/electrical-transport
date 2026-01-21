"""
Input: QD Data for samples with fraunhofers that show field switch
Output: Fraunhofer, Fitted Fraunhofer, Fit Parameters File

Last edited on October 5, 2020 by Swapna Sindhu Mishra
"""

folderpath = '' #File Location?
title = ''
sample='' #File name without the _Downsweep.dat/_Upsweep.dat
QD=4 #Which QD is being used?
downswitch=-30 #downswitch field value
upswitch=20 #upswitch field value

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy import optimize
os.chdir(folderpath) #changing working directory
downsweep = f'{sample}_Downsweep.dat' #change this according to your naming scheme
upsweep = f'{sample}_Upsweep.dat' #change this according to your naming scheme
downdata = np.loadtxt(downsweep) #loading downsweep data
updata = np.loadtxt(upsweep) #loading upsweep data
down_Ic_max=max(downdata[:,2]) #finding maximum Ic in downsweep
up_Ic_max=max(updata[:,2]) #finding maximum Ic in upsweep

if QD<3: #For conversion from Yixings
    down_Rn=5*9.461*downdata[1,3]*1.0006*10**(-6) #calculating downsweep resistance value
    up_Rn=5*9.461*updata[1,3]*1.0006*10**(-6) #calculating upsweep resistance value
else:    #For room temperature system
    down_Rn=downdata[1,3]*10**(3) #calculating downsweep resistance value
    up_Rn=updata[1,3]*10**(3) #calculating upsweep resistance value

down_IcRn=round(down_Ic_max*down_Rn, 1) #calculating raw downsweep IcRn
up_IcRn=round(up_Ic_max*up_Rn, 1) #calculating raw upsweep IcRn

#Plot and save raw fraunhofers with IcRn values in svg and png formats
plt.scatter(downdata[:,0],downdata[:,2], s=10, color='b', label=f'$I_C R_N$ = {down_IcRn} $\mu V$ $\downarrow$')
plt.scatter(updata[:,0],updata[:,2], s=10, color='r', label=f'$I_C R_N$ = {up_IcRn} $\mu V$')
plt.title(title, fontsize=15)
plt.figtext(.15, .8, f'{sample}')
plt.ylabel('Critical Current, $I_C$ (mA)', fontsize=15)
plt.xlabel('Field Strength, H (Oe)', fontsize=15)
plt.legend(loc='best', fontsize=10.5)
plt.savefig(f'{sample}.svg', bbox_inches='tight')
plt.savefig(f'{sample}.png', bbox_inches='tight', dpi=300)
plt.show()

"""
#This was written to automatically find values were switching happens (down/upswtich variable at top), but it's hit and miss depending on how sharp and conventional the switch is. Manual input works much better.
#Look for a sudden discontinuity in the data, assume switch is there
manual=1
if manual<1:
    downswitch=np.argmax(abs(np.ediff1d(downdata[:,2])))
    upswitch=np.argmax(abs(np.ediff1d(updata[:,2])))
else:
    downswitch=0
    upswitch=50
print(f'The switch in Downsweep is at {downswitch} Oe and Upswitch is at {upswitch} Oe')
"""

downH=downdata[:,0] #extract only H values from downdata
downrangeH=downH[downH > downswitch] #select downsweep H values bigger than switching field for fitting
ppp= len(downH)-len(downrangeH) #how many data points were removed from the entire range in the selection step above.
downrangeIc=downdata[ppp:,2] #selecting corresponding Ic values using the ppp variable
upH=updata[:,0] #extract only H values from updata
uprangeH=upH[upH < upswitch] #select upsweep H values smaller than switching field for fitting
uprangeIc=updata[:len(uprangeH),2] #selecting corresponding Ic values using the ppp variable


def fit_func_4(x,k,m,ic): #function needed for fitting fraunhofers. Variables explained in slide.
    return 2*ic*abs(scipy.special.jv(1,k*(x+m))/(k*(x+m)))
                                     
downparams,downparams_covariance=optimize.curve_fit(fit_func_4,downrangeH,downrangeIc,p0=[0.01, -50, 1]) #fitting downsweep data to function and extracting fit parameters
upparams,upparams_covariance=optimize.curve_fit(fit_func_4,uprangeH,uprangeIc,p0=[0.01, 50, 1]) #fitting upsweep data to function and extracting fit parameters

fit_file=open(f'{sample}_Airy_Fit_Parameters.dat', 'w') #saving downsweep fit parameters to file
fit_file.write(f'Down_Switch = {downswitch}\n')
fit_file.write(f'Down_k = {str(downparams[0])} +- {str(np.sqrt(np.diag(downparams_covariance))[0])} \n')
fit_file.write(f'Down_m = {str(downparams[1])} +- {str(np.sqrt(np.diag(downparams_covariance))[1])} \n')
fit_file.write(f'Down_Ic = {str(downparams[2])} +- {str(np.sqrt(np.diag(downparams_covariance))[2])} \n')
fit_file.write(f'Down_IcRn = {str(downparams[2]*down_Rn)} +- {str(np.sqrt(np.diag(downparams_covariance))[2]*down_Rn)}\n\n')

fit_file.write(f'Up_Switch = {upswitch}\n') #saving upsweep fit parameters to file
fit_file.write(f'Up_k = {str(upparams[0])} +- {str(np.sqrt(np.diag(upparams_covariance))[0])} \n')
fit_file.write(f'Up_m = {str(upparams[1])} +- {str(np.sqrt(np.diag(upparams_covariance))[1])} \n')
fit_file.write(f'Up_Ic = {str(upparams[2])} +- {str(np.sqrt(np.diag(upparams_covariance))[2])} \n')
fit_file.write(f'Up_IcRn = {str(upparams[2]*up_Rn)} +- {str(np.sqrt(np.diag(upparams_covariance))[2]*up_Rn)}')
fit_file.close()

downfit_IcRn=round(downparams[2]*down_Rn, 1) #downsweep IcRn value from the fitting
upfit_IcRn=round(upparams[2]*up_Rn, 1) #upsweep IcRn values from the fitting

#plotting and saving both raw and fitted fraunhofers for downsweep and upsweep in svg and png format
plt.title(title, fontsize=15)
plt.figtext(.15, .8, f'{sample}')
plt.ylabel('Critical Current, $I_C$ (mA)', fontsize=15)
plt.xlabel('Field Strength, H (Oe)', fontsize=15)
plt.scatter(downdata[:,0],downdata[:,2], s=1, color='b' )
plt.scatter(updata[:,0],updata[:,2], s=1, color='r')
plt.plot(downrangeH,fit_func_4(downrangeH,-downparams[0],downparams[1],downparams[2]),color='b', label=f'$I_C R_N$ = {downfit_IcRn} $\mu V$ $\downarrow$')
plt.plot(uprangeH,fit_func_4(uprangeH,upparams[0],upparams[1],upparams[2]), color='r', label=f'$I_C R_N$ = {upfit_IcRn} $\mu V$')
plt.legend(loc='best', fontsize=10.5)
plt.savefig(f'{sample}_Airy_Fit.svg', bbox_inches='tight')
plt.savefig(f'{sample}_Airy_Fit.png', bbox_inches='tight', dpi=300)
plt.show()
