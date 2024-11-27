import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as imgp
import scipy.io
from   scipy.io import loadmat
from   scipy.signal import convolve2d
import imageio

#Parameters
Re = 1500
tsim = 1
dt = 1e-5
nt = 13000  #int(np.round(tsim/dt))
 
Lx = 1
Ly = 1
Nx = 1024
Ny = 1024
dx = Lx/Nx
dy = Ly/Ny
 
# Campo de velocidad
u = np.zeros((Nx+1,Ny+2))   #eje x
v = np.zeros((Nx+2,Ny+1))   #eje y
uce = np.zeros((Nx,Ny))
vce = np.zeros((Nx,Ny))
 
uce = ( u[0:Nx,1:Ny+1] + u[1:Nx+1,1:Ny+1] )*0.5
vce = ( v[1:Nx+1,0:Ny] + v[1:Nx+1,1:Ny+1] )*0.5
 
# campo de presión
p = np.zeros((Nx,Ny))
 
W_h = np.array([
    [1],
    [-2],
    [1]
    ])
W_v = np.array([
    [1,-2,1]
    ])
 
prom_x = np.array([
    [1],
    [1]
    ])
 
prom_y = np.array([
    [1,1]
    ])
 
dvx = np.array([
    [-1],
    [1]
    ])
 
dvy = np.array([
    [-1,1]
    ])
 
# COPIAR HASTA AQUÍ
mat = loadmat('msgg.mat')   # cargar matriz msgg
msgg = mat['msgg']
 
# scipy.io.savemat('msgg_py.mat', {'matriz_python': msgg})
 
# Campo de temperatura
T  = np.zeros((Nx,Ny))
Tx = convolve2d(T,prom_x, mode='valid')
Ty = convolve2d(T,prom_y, mode='valid')
#print(Tx.shape)
#print(Ty.shape)
 
rho = 1e2
cal_p = 1
alpha1 = rho*cal_p
conduct = 1
Q = np.array((Nx,Ny))
Ga1 = np.array(imgp.imread('Cuerpo_1024.png'))
#plt.imshow(Ga1)
#plt.show()
 
body_u = 0*u
body_v = 0*v
body_u[1:Nx+1,1:Ny+1] = Ga1[:,:,0]
body_v[1:Nx+1,1:Ny+1] = Ga1[:,:,0]
Q = 1e5*Ga1[:,:,0]
p_bod = convolve2d(body_u[:,1:Ny+1],prom_x, mode='valid')
 
# Control temperatura
Sens = 0
Kp = 15
 
# Frontera campo velocidad
int_jet = 4
for k1 in range(0,Ny+2,1):
    u[0,k1]     = int_jet
    u[Nx,k1]    = int_jet
# scipy.io.savemat('u_py.mat', {'u_py': u})
 
 
video_frames = []
fps = 60
filename = 'simulacion9.mp4'
 
 
 
 
 
# NV con RK-4
for ii in range(0,nt,1):
    if (dt*ii)>0:
        u[0,k1]     = int_jet
        u[Nx,k1]    = int_jet
    else:
        u[0,k1]     = 0
        u[Nx,k1]    = 0
 
    # Laplacioano de campo de velocidad
    Lux = convolve2d(u[:,1:Ny+1],W_h, mode='valid')/(dx*dx)
    Luy = convolve2d(u[1:Nx,:],W_v,   mode='valid')/(dy*dy)
    Lvx = convolve2d(v[0:Nx+2,1:Ny],W_h, mode='valid')/(dx*dx)
    Lvy = convolve2d(v[1:Nx+1,0:Ny+1],W_v, mode='valid')/(dy*dy)
 
    # Interpolación al centro (ce) y a la esquina (co)
    uce = convolve2d(u[0:Nx+1,1:Ny+1],prom_x, mode='valid')/2
    uco = convolve2d(u[0:Nx+1,0:Ny+2],prom_y, mode='valid')/2
    vco = convolve2d(v[0:Nx+2,0:Ny+1],prom_x, mode='valid')/2
    vce = convolve2d(v[1:Nx+1,0:Ny+1],prom_y, mode='valid')/2
 
    # Producto para hallar aceleración conectiva
    uuce = uce*uce
    uvco = uco*vco
    vvce = vce*vce
 
    #print(uvco.shape)
    # Aceleración conectiva
    Nu_f = convolve2d(uuce,-dvx, mode='valid')
    Nu   = Nu_f/dx
    Nx_f = convolve2d(uvco[1:Nx,0:Ny+1],-dvy, mode='valid')
    Nu   = Nu + (Nx_f/dx)
 
    Nv_f = convolve2d(vvce,-dvy, mode='valid')
    Nv   = Nv_f/dy
    Ny_f = convolve2d(uvco[0:Nx+1,1:Ny],-dvx, mode='valid')
    Nv   = Nv + (Ny_f/dx)
 
    # Velocidad intermedia
    u[1:Nx,1:Ny+1] = u[1:Nx,1:Ny+1] + dt*(-Nu + (  (Lux+Luy)/Re  )  )
    v[1:Nx+1,1:Ny] = v[1:Nx+1,1:Ny] + dt*(-Nv + (  (Lvx+Lvy)/Re  )  )
 
    b_1 = convolve2d(u[0:Nx+1,1:Ny+1],-dvx, mode='valid')
    b_2 = convolve2d(v[1:Nx+1,0:Ny+1],-dvy, mode='valid')
    b   = (b_1/dx) + (b_2/dy)
    FFb = np.fft.fft2(b)
    u_F = msgg*FFb
    u_F = u_F*dx*dx
    ax1 = np.fft.ifft2(u_F)
    ax1 = ax1.real
    p   = ax1
 
    # Gradiente de la presion
    zx1 = convolve2d(p,-dvx, mode='valid')
    zx2 = convolve2d(p,-dvy, mode='valid')
 
    # Cálculo del campo de velocidad
    u[1:Nx,1:Ny+1] = ( u[1:Nx,1:Ny+1] - (zx1/dx)*(1-body_u[1:Nx,1:Ny+1]))
    v[1:Nx+1,1:Ny] = ( v[1:Nx+1,1:Ny] - (zx2/dy)*(1-body_v[1:Nx+1,1:Ny]))
 
    # Creación video de salida
    uce = convolve2d(u[0:Nx+1,1:Ny+1],prom_x, mode='valid')/2
    vce = convolve2d(v[1:Nx+1,0:Ny+1],prom_y, mode='valid')/2
 
    # Cálculo de la ecuacion de temperatura
    LTx = convolve2d(T[0:Nx,1:Ny-1],W_h, mode='valid')/(dx*dx)
    LTy = convolve2d(T[1:Nx-1,0:Ny],W_v, mode='valid')/(dy*dy)
 
    fTx = convolve2d(Tx[0:Nx-1,1:Ny-1],dvx, mode='valid')/dx
    fTy = convolve2d(Ty[1:Nx-1,0:Ny-1],dvy, mode='valid')/dy
 
    cont_temp = np.tanh(0.01*((25-np.mean(T[660:670,510:520]))))*Kp
    cont_temp = cont_temp*(cont_temp>=0)
 
    T[1:Nx-1,1:Ny-1] = ((dt*conduct)/alpha1)*(LTx+LTy)+dt*uce[1:Nx-1,1:Ny-1]*fTx+dt*vce[1:Nx-1,1:Ny-1]*fTy+(dt/alpha1)*Q[1:Nx-1,1:Ny-1]*cont_temp+T[1:Nx-1,1:Ny-1]
    Tx = convolve2d(T,prom_x, mode='valid')
    Ty = convolve2d(T,prom_y, mode='valid')
    print(ii)
    mean_field_vel = np.sqrt(uce*uce+vce*vce)
    mean_field_vel = (mean_field_vel - np.min(mean_field_vel))/(np.max(mean_field_vel)-np.min(mean_field_vel))
    frame = mean_field_vel
    video_frames.append(frame)
    if (ii%10)==1:
        #mean_field_vel = np.sqrt(uce*uce+vce*vce)
        plt.clf()
        plt.imshow(mean_field_vel)
        plt.draw()
        plt.pause(0.1)
        print(ii)
    
    #print(LTx.shape)
    #print(LTy.shape)
    #scipy.io.savemat('FFb_py.mat', {'FFb_py': FFb})
    #scipy.io.savemat('b_py.mat', {'b_py': b})
    #scipy.io.savemat('u_F_py.mat', {'u_F_py': u_F})
    #scipy.io.savemat('p_py.mat', {'p_py': p})
    #scipy.io.savemat('LTx_py.mat', {'LTx_py': LTx})
    #scipy.io.savemat('LTy_py.mat', {'LTy_py': LTy})
    #break
    '''
    print([uce.shape,vce.shape,uco.shape,vco.shape])
    print([np.max(uce),np.max(vce),np.max(uco),np.max(vco)])
    break
    '''
print("fin")
imageio.mimsave(filename, video_frames, fps=fps)
