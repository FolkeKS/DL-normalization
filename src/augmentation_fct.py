
import numpy as np

def flip_vert(X):
    """
        flip north <-> south (up, down)
    """
    alphas_n = np.flipud(X[0,:,:])
    alphas_s = np.flipud(X[1,:,:])
    alphas_e = np.flipud(X[2,:,:])
    alphas_w = np.flipud(X[3,:,:])
    w        = np.flipud(X[4,:,:])
    
    X[0,:,:] = alphas_s
    X[1,:,:] = alphas_n
    X[2,:,:] = alphas_e
    X[3,:,:] = alphas_w
    X[4,:,:] = w
    return X


def flip_hor(X):
    """
        flip west <-> east (left, right)
    """
    alphas_n = np.flipud(X[0,:,:])
    alphas_s = np.flipud(X[1,:,:])
    alphas_e = np.flipud(X[2,:,:])
    alphas_w = np.flipud(X[3,:,:])
    w        = np.flipud(X[4,:,:])
    
    X[0,:,:] = alphas_n
    X[1,:,:] = alphas_s
    X[2,:,:] = alphas_w
    X[3,:,:] = alphas_e
    X[4,:,:] = w
    return X

def rot_90(X):

    alphas_n = np.rot_90(X[0,:,:], k=1, axes=(0,1))
    alphas_s = np.flipud(X[1,:,:], k=1, axes=(0,1))
    alphas_e = np.flipud(X[2,:,:], k=1, axes=(0,1))
    alphas_w = np.flipud(X[3,:,:], k=1, axes=(0,1))
    w        = np.flipud(X[4,:,:], k=1, axes=(0,1))
    
    X[0,:,:] = alphas_e
    X[1,:,:] = alphas_w
    X[2,:,:] = alphas_s
    X[3,:,:] = alphas_n
    X[4,:,:] = w
    return X


def rot_180(X):

    alphas_n = np.rot_90(X[0,:,:], k=2, axes=(0,1))
    alphas_s = np.flipud(X[1,:,:], k=2, axes=(0,1))
    alphas_e = np.flipud(X[2,:,:], k=2, axes=(0,1))
    alphas_w = np.flipud(X[3,:,:], k=2, axes=(0,1))
    w        = np.flipud(X[4,:,:], k=2, axes=(0,1))
    
    X[0,:,:] = alphas_s
    X[1,:,:] = alphas_n
    X[2,:,:] = alphas_w
    X[3,:,:] = alphas_e
    X[4,:,:] = w
    return X


def rot_180(X):

    alphas_n = np.rot_90(X[0,:,:], k=1, axes=(1,0))
    alphas_s = np.flipud(X[1,:,:], k=1, axes=(1,0))
    alphas_e = np.flipud(X[2,:,:], k=1, axes=(1,0))
    alphas_w = np.flipud(X[3,:,:], k=1, axes=(1,0))
    w        = np.flipud(X[4,:,:], k=1, axes=(1,0))
    
    X[0,:,:] = alphas_w
    X[1,:,:] = alphas_e
    X[2,:,:] = alphas_n
    X[3,:,:] = alphas_s
    X[4,:,:] = w
    return X