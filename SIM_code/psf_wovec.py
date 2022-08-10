import numpy as np
import matplotlib.pyplot as plt
from pylab import *

N= 101 #Number of grid along x or y axis
na=1 #Numerical aperture
Light_lambda_nm=633 #Light wavelength(m)
Light_lambda=Light_lambda_nm*10**(-9) #Light wavelength(m)
k=2*np.pi/Light_lambda #Wave vector
i=np.arange(N)-int(N/2)
print(i)
kxyz=[[[0]*3 for i in range(N)] for i2 in range(N)]
rxy=[[[0]*2 for i in range(N)] for i2 in range(N)]
ixy=[[0 for i in range(N)] for i2 in range(N)]
ucossin=[[[0]*2 for i in range(N)] for i2 in range(N)]
U=zeros((N,N),dtype=complex)
a=zeros((N,N),dtype=complex)
kxyz_set=[]
#n_lambda=4 #Number of unit of wavelength at x-y surface
n_xyz=N*10 #Number of x or y distance (nm)
count=0
a0=1
z=0
print("1 pix. = " + str(n_xyz/N) + "nm")

for n in range(N):
    for m in range(N):
        kxyz[n][m][0]=2*k*i[n]/N
        kxyz[n][m][1]=2*k*i[m]/N
        #rxy[n][m][0]=2*n_lambda*lanmba*i[n]/N
        #rxy[n][m][1]=2*n_lambda*lanmba*i[m]/N
        rxy[n][m][0]=n_xyz*10**(-9)*i[n]/N
        rxy[n][m][1]=n_xyz*10**(-9)*i[m]/N

        c=kxyz[n][m][0]**2+kxyz[n][m][1]**2
        if c<=(k*na)**2:
            a[n][m]=a0
            kxyz[n][m][2]=np.sqrt(k**2-c)
            kxyz_set.append([kxyz[n][m][0],kxyz[n][m][1],kxyz[n][m][2]])
            count=count+1

for l in range(count):
    for n in range(N):
        for m in range(N):
            theta=(kxyz_set[l][0]*rxy[n][m][0]+kxyz_set[l][1]*rxy[n][m][1]+kxyz_set[l][2]*z)
            U[n][m]=U[n][m]+a[n][m]*np.exp(complex(0, theta))
            
U=np.power(np.abs(U)/(a0*count),2)

x=[]
y=[]
for i in range(N):
    x.append(rxy[i][0][0])
    y.append(rxy[0][i][1])
#print(x)
X, Y = meshgrid(x, y)
pcolor(X, Y, U, cmap="hot")
colorbar()
#plt.axes().set_aspect('equal', 'datalim')
#plt.axes().set_aspect('equal')
plt.gca().set_aspect('equal')
plt.savefig('Intensity_Na'+str(na)+'_wavelength'+str(Light_lambda_nm)+'nm.png')
show()

np.savetxt('Intensity_Na'+str(na)+'_wavelength'+str(Light_lambda_nm)+'nm.csv', U, delimiter=",")
