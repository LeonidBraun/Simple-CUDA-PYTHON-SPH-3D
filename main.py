import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import time
from pyevtk.hl import pointsToVTK

file = open(r'ker_rho.cu','r')
ker_rho = cp.RawKernel(file.read(), 'ker_rho')
file.close()

file = open(r'ker_dv.cu','r')
ker_dv = cp.RawKernel(file.read(), 'ker_dv')
file.close()

def grid_ind(ind, n,q):
    
    m = cp.arange(n[0]*n[1]*n[2])
    ind[m,0] = cp.searchsorted(q, m, side='left')
    ind[m,1] = cp.searchsorted(q, m, side='right')
    return

def sort_grid(x,q,h):
    X = cp.floor((x-cp.min(x,0)+1e-6)/(1.1*h)).astype(cp.int32)
    n = tuple(cp.asnumpy(cp.ceil(0.7+cp.max(X,0)).astype(cp.int32)))
    q = n[1]*n[2]*X[:,0]+n[2]*X[:,1]+X[:,2]
    
    per = cp.argsort(q)
    x = x[per,:]
    q = q[per]

    ind = cp.zeros( (n[0]*n[1]*n[2],2),dtype=cp.int32)
    
    grid_ind(ind,n,q)
    
    return n, q, per, ind, x

def save(it,x,v,rho,pId,q,folder):
    pointsToVTK('./'+folder+'/test_'+str(it),cp.asnumpy(x[:,0]),cp.asnumpy(x[:,1]),cp.asnumpy(x[:,2]),
        data =  {       'P_ID':         cp.asnumpy(pId),
                        'Q':            cp.asnumpy(q),
                        'RHO':          cp.asnumpy(rho),
                        'VEL':          (cp.asnumpy(v[:,0]),cp.asnumpy(v[:,1]),cp.asnumpy(v[:,2])),
                } )

def initialize():
    N = 100*100*100
    N=10*10*10*2000
    a=1
    b=1
    c=1
    dx = np.cbrt((a*b*c)/N)
    nx = a/dx
    ny = b/dx
    nz = c/dx
    N = nx*ny*nz
    print(nx)

    he = (cp.mgrid[:nx,:ny,:nz]+0.5)*dx
    x = he.reshape((3,-1)).T - cp.array([0,0,0])[None,:]
    x = x.astype(cp.float32)

    h = np.float32(dx * 3)

    return x, h

folder = 'Output'

x, h = initialize()
v = x*0+0
rho = x[:,0]*0+0

dv = v*0+0

pId = cp.arange(x.shape[0]).astype(cp.int32)
q = cp.zeros(x.shape[0]).astype(cp.int32)

# warmup
n, q, per, ind, x = sort_grid(x,q,h)
v = v[per]
rho = rho[per]
pId = pId[per]

ker_rho( n, (64,), ( rho, x, ind, h ) )
# get stated
save(0,x,v,rho,pId,q,folder)

dt = 0.01*h
for i in range(200):
    
    for j in range(10):
        n, q, per, ind, x = sort_grid(x,q,h)
        v = v[per]
        rho = rho[per]
        pId = pId[per]

        for k in range(20):
            ker_rho( n, (64,), ( rho, x, ind, h ) )
            ker_dv( n, (64,), ( dv, x, v,rho, ind, h ) )

            v += dt*dv
            x += dt*v

    ker_rho( n, (64,), ( rho, x, ind, h ) )
    save(i+1,x,v,rho,pId,q,folder)
    print('Save of: ',i+1)
    print(cp.sum(v,0))