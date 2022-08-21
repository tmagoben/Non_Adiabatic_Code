import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.special as scl
import numpy.matlib as mat
import scipy.fftpack as fft
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Slider, Button

##Split-Operator Yarkony 3D High-Symmetry Spin Orbit###

# Constants

ω = np.array([1,1,1]) #Frequency for each coordinate


gs = np.array([0,0,0]) #Inital Wavepacket shifts

g = 0.0835 # g-vector:(g_j-g_i)/2

h = 0.0430

h_i = 0.000233

s_y = 0# s_y vector (g_jy+g_iy)/2

s_x = 0.05 # s_x vector (g_jx+g_ix)/2

s_z = 0.125 # s_y vector (g_jz+g_iz)/2


#Location of conical intersection: (0,0,0)--> you can translate to coordinates in article if desired

### If you want to switch what surface WP is propagating on, change here###
iS = 0 # Intial starting state

x0c = gs[0]

y0c = gs[1]

z0c = gs[2]

##intial Momenta

kIx = 0

kIz = 0

kIy = 0

##Set up a grid #Lets see how this goes

Mx = 64*2

My = 64*2

Mz = 64*2

#Number of states


N = 2

#Number of time steps


Tsteps = 30

dt = 0.05

# Grid Lengths

Lx = 10

Ly = 10

Lz = 10

LxT = Lx*2

LyT = Ly*2

LzT = Lz*2

#Grid of M points

x0 = np.linspace(-Lx,Lx,Mx)

y0 = np.linspace(-Ly,Ly,My)

z0 = np.linspace(-Lz,Lz, Mz)

#Parameters
#k0[1xM] = Grid of M momenta points from 0->L

k0x = np.linspace(-Mx*np.pi/LxT,Mx*np.pi/LxT-2*np.pi/LxT,Mx)

k0y = np.linspace(-My*np.pi/LyT,My*np.pi/LyT-2*np.pi/LxT,My)

k0z = np.linspace(-Mz*np.pi/LzT,Mz*np.pi/LzT-2*np.pi/LzT,Mz)



##Properties

mex  = np.zeros((N,Tsteps))#<x>
mex2 = np.zeros((N,Tsteps))#<x^2>
mez = np.zeros((N,Tsteps))#<z>
mey = np.zeros((N,Tsteps))#<y>
mepx = np.zeros((N,Tsteps))#mep[n+1, Tsteps] momenta on each diabate
meabs = np.zeros((N,Tsteps))#mabs[N+1, Tsteps] = population on each diabate
mepx2 = np.zeros((N,Tsteps))#mep[N+1, Tsteps] momenta^2 on each diabate
mepz = np.zeros((N,Tsteps))
mepy = np.zeros((N,Tsteps))

x0op = (np.tile(x0,(Mz,1))).T

y0op = (np.tile(y0,(Mz,1))).T

z0op = np.tile(z0,(Mx,1))

k0xop = (np.tile(k0x,(Mz,1))).T

k0yop = (np.tile(k0x,(My,1))).T

k0zop = (np.tile(k0z,(Mz,1)))

##Inital wavefunction: 2D gaussian
ψ_0 = np.zeros((N,Mx*My),dtype = 'complex')
σ_x = np.sqrt(2/ω[0])
σ_y = np.sqrt(2/ω[1])
σ_z = np.sqrt(2/ω[1])
temp = np.tile(np.exp(-((z0-z0c)/(σ_z))**2)*np.exp(1.j*kIz*z0),(Mx,1))

temp_y = np.tile(np.exp(-((y0-y0c)/σ_y)**2)*np.exp(1.j*kIy*y0),(Mz,1)).T

temp_1 = temp_y*temp*(np.tile(np.exp(-((x0-x0c)/σ_x)**2)*np.exp(1.j*kIx*x0),(Mz,1)).T)


##This is where python and MATLAB vary, we want to populate the first
##state with ort temp matrix, I
ψ_0[iS,:] = temp_1.reshape(1,Mx*Mz)##Gaussian wavepacket on adatom Diabatic

##
ψ_0[iS,:] = ψ_0[iS,:]/np.sqrt(ψ_0[iS,:]@ψ_0[iS,:].T)##Normalized wavefunction

#Now we need to calculate the propagators

##Kinetic T = exp(i*hbark^2.2mdt/hbar)

TP = np.exp(-1.j*(np.tile((k0x**2)/2,(Mz,1)).T+np.tile((k0y**2)/2,(Mz,1))+np.tile((k0z**2)/2,(Mx,1)))*dt)

T = np.tile((k0x*k0x)/2,(Mz,1)).T +np.tile((k0y*k0y)/2,(Mz,1)).T + np.tile((k0z*k0z)/2,(Mx,1))

ψ_0

##Potential Energy propagator V
VDI = np.zeros((2,2), dtype = 'complex')

D2A1 = np.zeros((2,Mx*Mz), dtype = 'complex')
D2A2 = np.zeros((2,Mx*Mz), dtype = 'complex')
V1 = np.zeros((2,Mx*Mz), dtype = 'complex')
V2 = np.zeros((2,Mx*Mz), dtype = 'complex')
VP1 = np.zeros((2,Mx*Mz), dtype = 'complex')
VP2 = np.zeros((2,Mx*Mz), dtype = 'complex')
for ig in range(Mz):
    for jg in range(Mx):
        for kg in range(My):
            z = z0[ig]
            y = y0[kg]
            x = x0[jg]
###Diabtic Matrix###
            VDI[0,0] = -g*(z)+s_x+s_y+s_z
            VDI[1,1] = g*(z)+s_x+s_y+s_z
            VDI[0,1] = h*(x)+1.j*h_i*y
            VDI[1,0] = np.conj(VDI[0,1])
###Adiabatization###
            [VDt, U] = np.linalg.eigh(VDI)

            VDt = np.array(VDt).reshape(2,1)
            VDt = np.diagflat(VDt)
            UUdVP = U@sp.linalg.expm(-1.j*dt*VDt)@U.T
            np.shape(UUdVP)

            V = U@VDt@U.T
            ixz = jg+(ig)*Mx
            D2A1[:, ixz] =  np.conj((U[:,0]))
            D2A2[:, ixz] =  np.conj((U[:,1]))
            V1[:, ixz] = VDI[:,0]
            V2[:, ixz] = VDI[:,1]
            VP1[:,ixz] = UUdVP[:,0]
            VP2[:,ixz] = UUdVP[:,1]


VDI


ψ = ψ_0 #Initialization of wavepacket

tR = np.linspace(0,dt*Tsteps,Tsteps)

ek = np.zeros((Tsteps, N), dtype = 'complex')

e = np.zeros((Tsteps, 1), dtype = 'complex')

ev = np.zeros((Tsteps,1), dtype = 'complex')

norm = np.zeros((Tsteps,1), dtype = 'complex')

meabs_adi = np.zeros((Tsteps,N), dtype ='complex').T
#print(psi) Uncheck to make sure everything is proper
for t in range(Tsteps):
    ψ  = VP1*np.vstack((ψ[0,:],ψ[0,:])) +  VP2*np.vstack((ψ[1,:],ψ[1,:]))
#### If errors propagate, look here
##T propgators

    for n in range(N):
        temp2 = ψ[n,:]
        temp3 = temp2.reshape(Mx,Mz)
        temp4 = fft.fftshift(fft.fft2(temp3))
        ek[t,n] = np.real(np.sum(np.sum(np.conj(temp4)*T*temp4))/np.sum(np.sum(np.conj(temp4)*temp4)))
        temp5 = temp4*TP#Kinetic Propagator
        temp6 = fft.ifft2(fft.fftshift(temp5))
        ψ[n,:] = temp6.reshape(1,Mx*Mz)

####Now we check the energy conservation####
    ###ek(t)+ev(t) = constant
    ket = V1*np.vstack((ψ[0,:],ψ[0,:])) +  V2*np.vstack((ψ[1,:],ψ[1,:]))

    ev[t] = np.real(np.sum(np.sum((np.conj(ψ)*ket))))/np.real(np.sum(np.conj(ψ)*ψ))

    ψ_adi = D2A1*np.vstack((ψ[0,:],ψ[0,:])) + D2A2*np.vstack((ψ[1,:], ψ[1,:]))
    meabs_adi[:,t] = np.sum(np.conj(ψ_adi)*ψ_adi,1)
    meabs[:,t] = np.sum(np.conj(ψ)*ψ,1) ## population on each dibat
    e[t] = ev[t] + ek[t,0]*meabs[0,t] + ek[t,1]*meabs[1,t]
    norm[t] = np.real(np.sum(np.sum(np.conj(ψ)*ψ_0)))
    ψ_0 = ψ

print(e[:50])## if you want to display all values remove(:10)



##Population Dynamics###
fig1 = plt.figure(figsize = (10,10))
plt.plot(tR,np.real(meabs[0,:]), label = "Exact")
plt.plot(tR,np.real(meabs_adi[0,:]), label = "Adi")
plt.xlabel('Time')
plt.ylabel(' $S_{0}-SOC$ Population')
plt.legend()
plt.show()
#fig1.savefig('S_0_SOC_Exact_Vs_Adi_population.jpg')

fig2 = plt.figure(figsize = (10,10))
plt.plot(tR,np.real(meabs[1,:]), label = "Exact")
plt.plot(tR,np.real(meabs_adi[1,:]), label = "Adi")
plt.xlabel('Time')
plt.ylabel(' $S_{1}-SOC$ Population')
plt.legend()
plt.show()
#fig2.savefig('S_1_SOC_Exact_Vs_Adi_population.jpg')

###Nuclear Wavepacket###

##S_0χ
fig3 = plt.figure(figsize = (10,10))
plt.imshow(np.abs(ψ[0,:].reshape(Mx,Mz)))
plt.xlabel('x')
plt.ylabel('z')
plt.show()
#fig3.savefig('S_0_SOC_Nuclear_Wavepacket.jpg')

fig4 = plt.figure(figsize = (10,10))
X,Y = np.meshgrid(x0,z0)
ax = plt.axes(projection='3d')
surf =ax.plot_surface(X,Y,np.abs(ψ[0,:].reshape(Mx,Mz)),cmap = cm.inferno)
ax.view_init(25, 50)
plt.xlabel('x')
plt.ylabel('y')
fig4.colorbar(surf, shrink=0.5, aspect=5)
#fig4.savefig('S_0_SOC_Nuclear_Wavepacket_Surface.jpg')


##S_1χ
fig5 = plt.figure(figsize = (10,10))
plt.imshow(np.abs(ψ[1,:].reshape(Mx,Mz)),label = "S_1")
plt.xlabel('x')
plt.ylabel('z')
plt.show()
#fig5.savefig('S_1_SOC_Nuclear_Wavepacket.jpg')


fig6 = plt.figure(figsize = (10,10))
X,Y = np.meshgrid(x0,z0)
ax = plt.axes(projection='3d')
surf =ax.plot_surface(X,Y,np.abs(ψ[1,:].reshape(Mx,Mz)),cmap = cm.inferno)
ax.view_init(25, 50)
plt.xlabel('x')
plt.ylabel('y')
fig6.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
#fig6.savefig('S_1_SOC_Nuclear_Wavepacket_Surface.jpg')




###Norm Conservation###
fig7 = plt.figure(figsize = (10,10))
plt.plot(tR,np.abs(norm), label = 'Norm')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Norm')
plt.show()
#fig7.savefig('Norm_SOC_Conservation.jpg')


###Energy Conservation###
fig8 = plt.figure(figsize = (10,10))
plt.plot(tR,np.abs(e), label = 'Energy')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy')
plt.show()
#fig8.savefig('Energy_SOC_Conservation.jpg')
