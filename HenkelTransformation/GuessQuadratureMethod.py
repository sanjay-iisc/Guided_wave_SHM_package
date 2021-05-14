from numpy import *

# Recursive generation of the Legendre polynomial of order n
def f(x):
    return 1/x

def Legendre(n,x):
	x=array(x)
	if (n==0):
		return x*0+1.0
	elif (n==1):
		return x
	else:
		return ((2.0*n-1.0)*x*Legendre(n-1,x)-(n-1)*Legendre(n-2,x))/n
# def interpolation_Langrange():

if __name__ =='__main__':
    L1 = Legendre(2, x)
    L2 = Legendre(2, x)
