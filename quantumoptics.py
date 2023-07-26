import numpy as np
import qutip as qt
from scipy.sparse.linalg import eigs as sparse_eigs
from scipy.linalg import eigh as dense_eigs
from numpy.linalg import norm as np_norm

__all__ = ['vec_to_dm','vec_to_operator','decompose_L','decompose_rho','steadystate']



def vec_to_dm(vec, dims, shape): 
    _temp = vec.reshape(shape, order = 'F') # reshape matrix from array
    mat = 0.5*(_temp+np.conjugate(_temp.T))  
    mat = mat/np.trace(mat) 
    dm = qt.Qobj(mat, dims=dims, type='oper')
    return dm

def vec_to_operator(vec, dims, shape): 
# the same function as above but without normalization: suitable for creating traceless operators from eigenvectors to the liouvillian
    _temp = vec.reshape(shape, order = 'F') # reshape matrix from array
    mat = 0.5*(_temp+np.conjugate(_temp.T))  
    dm = qt.Qobj(mat, dims=dims, type='oper')
    return dm

def decompose_L(L,k=5,sigma=1e-10,which='LM'):    
    evals, evecs = sparse_eigs(L.data, k=5, sigma = 1e-5, which = 'LM')
    evecs = [evecs[:,i] for i in range(0,len(evals))]
    
    # Calculate normalized and phase fixed vectors: vec = vec/norm*phase
    # phase = (np.abs(vec[phase_fix]) / vec[phase_fix])
    phase_fix = 0
    vecs = np.array([vec/np_norm(vec)*(np.abs(vec[phase_fix]) / vec[phase_fix]) for vec in evecs])
    return evals, vecs

def steadystate(L,k=1,sigma=1e-15,which='LM'):
    # Sparse solver for finding the steady states of the LIouvillian L
    dims = (L.dims)[0]
    shape = (int(np.sqrt((L).shape)[0]),int(np.sqrt((L).shape)[0]))
    evals, evecs = sparse_eigs(L.data, k=5, sigma = 1e-5, which = 'LM')
    ss = np.transpose(evecs)[0]
    ss = vec_to_dm(ss, dims, shape)
    return ss

def decompose_rho(operator):
    p = qt.Qobj(dims = operator.dims, shape = operator.shape, type = 'oper')
    m = qt.Qobj(dims = operator.dims, shape = operator.shape, type = 'oper')
    
    matrix = operator.full()
    ee, vv = dense_eigs(matrix)
    ee = np.real(ee)
    idx_p = (np.where(ee>0))[0]
    idx_m = (np.where(ee<0))[0]
    
    for i in idx_p:
        op = qt.Qobj(np.outer(vv[:,i],np.conjugate(vv[:,i])),dims=p.dims, shape=p.shape)
        p += ee[i] * op
    for i in idx_m:
        op = qt.Qobj(np.outer(vv[:,i],np.conjugate(vv[:,i])),dims=p.dims, shape=p.shape)
        m += -ee[i] * op
    
    if np.trace(m) == 0:
        return p/np.trace(p), m

    p = p/np.trace(p)
    m = m/np.trace(m)
    return p, m 
