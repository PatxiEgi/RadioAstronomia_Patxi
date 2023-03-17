# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:23:16 2023

@author: Patxi
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# En los siguientes archivos irán los resultados.

archivo=open('Valores_tau.txt','w')
archivo2=open('Residuals.txt','w')
archivo3=open('Temperatura_Receptor.txt','w')
archivo4=open('Temperatura_sist_y_sigma.txt','w')

# Leemos los archivos y ponemos una variable a cada dato.

hdul = fits.open('QUIJOTE-DIP270D-180120-0257.fits')
data=hdul[1].data

JD=data.field('JD')[0]
Tbrillo_11=data.field('DATA11')[0]
Tbrillo_13=data.field('DATA13')[0]
Tbrillo_17=data.field('DATA17')[0]
Tbrillo_19=data.field('DATA19')[0]

Azim_11_13=data.field('AZ_11_13')[0]
Azim_17_19=data.field('AZ_17_19')[0]
Elev_11_13=data.field('EL_11_13')[0]
Elev_17_19=data.field('EL_17_19')[0]


T_atm=262 #Kelvin
T_CMB=2.7 #Kelvin

# Creamos los siguientes variables para dibujar las líneas.

Y_11=Tbrillo_11/T_atm
Y_13=Tbrillo_13/T_atm
Y_17=Tbrillo_17/T_atm
Y_19=Tbrillo_19/T_atm

X_11=X_13=1/(np.cos(np.pi*(90-Elev_11_13)/180))
X_17=X_19=1/(np.cos(np.pi*(90-Elev_17_19)/180))


# Determinamos los parámetros y estilo de la figura.

plt.figure(1,figsize=(20,10))

plt.style.use('bmh')
plt.title('Opacidad atmosférica respecto a la masa de aire ',size=30)
plt.xlabel('sec(z)',size=25)
plt.ylabel(r'$T_b$/$T_{atm}$',size=25)
plt.minorticks_on()

plt.scatter(X_11,Y_11,label='11GHz')
plt.scatter(X_13,Y_13,label='13GHz')
plt.scatter(X_17,Y_17,label='17GHz')
plt.scatter(X_19,Y_19,label='19GHz,')


X11=[]
Y11=[]
X13=[]
Y13=[]
X17=[]
Y17=[]
X19=[]
Y19=[]
for i in range (len(X_11)):
    X11.append(X_11[i][0])
    Y11.append(Y_11[i][0])
    X13.append(X_13[i][0])
    Y13.append(Y_13[i][0])
    X17.append(X_17[i][0])
    Y17.append(Y_17[i][0])
    X19.append(X_19[i][0])
    Y19.append(Y_19[i][0])


# Hacemos la regresión lineal para sacar los valores, y los dibujamos.

z11= np.polyfit(X11, Y11, 1,full='True')
z13= np.polyfit(X13, Y13, 1,full='True')
z17= np.polyfit(X17, Y17, 1,full='True')
z19= np.polyfit(X19, Y19, 1,full='True')


plt.plot(X_11,z11[0][1]+X_11*z11[0][0],color='black',ls='--',label='Ajustes')
plt.plot(X_13,z13[0][1]+X_13*z13[0][0],color='black',ls='--')
plt.plot(X_17,z17[0][1]+X_17*z17[0][0],color='black',ls='--')
plt.plot(X_19,z19[0][1]+X_19*z19[0][0],color='black',ls='--')

plt.legend(labelcolor='blue',fontsize=20)

# Calculamos las temperaturas del receptor.



T_recep11=z11[0][1]*T_atm-T_CMB
T_recep13=z13[0][1]*T_atm-T_CMB
T_recep17=z17[0][1]*T_atm-T_CMB
T_recep19=z19[0][1]*T_atm-T_CMB

# Calculamos la temperatura del sistema para cada valor de elevación, a cada frecuencia.

T_sist_11= np.mean(T_recep11+T_CMB+T_atm*z11[0][0]*np.array(X11))
T_sist_13= np.mean(T_recep13+T_CMB+T_atm*z13[0][0]*np.array(X13))
T_sist_17= np.mean(T_recep17+T_CMB+T_atm*z17[0][0]*np.array(X17))
T_sist_19= np.mean(T_recep19+T_CMB+T_atm*z19[0][0]*np.array(X19))

sigma_11=T_sist_11/np.sqrt(2*10**9*0.06)
sigma_13=T_sist_13/np.sqrt(2*10**9*0.06)
sigma_17=T_sist_17/np.sqrt(2*10**9*0.06)
sigma_19=T_sist_19/np.sqrt(2*10**9*0.06)




# Escribimos los resultados en 4 archivos.

archivo.write('    Frecuencia   '+' Opacidad atmosférica en el cénit    '+' Ordenada de origen')
archivo.write('\n')
archivo.write('    11GHz'+'               '+str(z11[0][0])+'             '+ str(z11[0][1]))
archivo.write('\n')
archivo.write('    13GHz'+'               '+str(z13[0][0])+'             '+ str(z13[0][1]))
archivo.write('\n')
archivo.write('    17GHz'+'               '+str(z17[0][0])+'             '+ str(z17[0][1]))
archivo.write('\n')
archivo.write('    19GHz'+'               '+str(z19[0][0])+'             '+ str(z19[0][1]))


archivo2.write('    Frecuencia   '+'      RMS')
archivo2.write('\n')
archivo2.write('    11GHz'+'               '+str(z11[1]))
archivo2.write('\n')
archivo2.write('    13GHz'+'               '+str(z13[1]))
archivo2.write('\n')
archivo2.write('    17GHz'+'               '+str(z17[1]))
archivo2.write('\n')
archivo2.write('    19GHz'+'               '+str(z19[1]))
archivo2.write('\n')

archivo3.write('    Frecuencia   '+'      Temperatura del receptor [K]')
archivo3.write('\n')
archivo3.write('    11GHz'+'               '+str(T_recep11))
archivo3.write('\n')
archivo3.write('    13GHz'+'               '+str(T_recep13))
archivo3.write('\n')
archivo3.write('    17GHz'+'               '+str(T_recep17))
archivo3.write('\n')
archivo3.write('    19GHz'+'               '+str(T_recep19))
archivo3.write('\n')

archivo4.write('    Frecuencia   '+' Temperatura del sistema    '+' Desviación usando ec. radiometro ideal')
archivo4.write('\n')
archivo4.write('    11GHz'+'               '+str(T_sist_11)+'             '+ str(sigma_11))
archivo4.write('\n')
archivo4.write('    13GHz'+'               '+str(T_sist_13)+'             '+ str(sigma_13))
archivo4.write('\n')
archivo4.write('    17GHz'+'               '+str(T_sist_17)+'             '+ str(sigma_17))
archivo4.write('\n')
archivo4.write('    19GHz'+'               '+str(T_sist_19)+'             '+ str(sigma_19))




archivo.close()
archivo2.close()
archivo3.close()
archivo4.close()




    








