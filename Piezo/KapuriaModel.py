import numpy as np
import matplotlib.pyplot as plt
# plt.style.use(['science', 'ieee', 'grid'])
V=1
a=10e-3
alpha=4
hp =0.25e-3
ha=40e-6
Ga=0.5e9
Ez=-V/hp
d31=-265e-12
d32=-265e-12
Y1=66.67e9
Y2=66.67e9
s11=1/Y1
s22=1/Y2
v21=0.29
v12=0.29
s12=-v21/Y1
Beta=ha/Ga
s11_bar = s11-(s12**2/s22)
d31_bar = d31- (d32*s12/s22)
###Elastic plates
E1=70.30e9
s11_s= 1/E1
h =1.5e-3

tau1 = (alpha*s11_s)/(Beta*h)
tau2 = (s11_bar)/(Beta*hp)

tau =np.sqrt(tau1+tau2)

x = np.linspace(-a/2 , a/2 , 1000, endpoint=True)
sigma1 = (d31_bar*Ez*np.sinh(tau*x))

sigma2 = Beta*tau*np.cosh(tau*a/2)

Y0=70.3e9
d0=-d31
sigma =(sigma1/sigma2)#*a/(Y0*V*d0)
plt.figure()
plt.plot(x/a, sigma)
# plt.ylim([-12,12])

plt.show()
