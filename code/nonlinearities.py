import numpy as np

NONPARAMETRICAL_TRANSROMATIONS = ['linear', 'p2', 'p3', 'p4', 's2', 's3', 's4']


def f(x, kind='linear', params=None):
    if kind == 'linear':
        return x
    elif kind == 'g1':
        return np.sign(x) * np.exp(params[0]) * (np.exp(params[1] * np.abs(x)) - 1)
    elif kind == 'g2':
        return np.sign(x) * np.exp(params[0]) * np.sqrt(np.abs(x))
    elif kind == 'g3':
        return np.sign(x) * np.exp(params[0]) * (np.exp(params[1] * np.sqrt(np.abs(x))) - 1)
    elif kind == 'g4':
        return np.sign(x) * np.exp(params[0]) * (np.exp(params[1] * np.abs(x) ** (1./3)) - 1)
    elif kind == 'g5':
        return np.sign(x) * params[0] * (np.exp(np.abs(x)) - 1)
    elif kind in ['p2', 'p3', 'p4']:
        power = int(kind[-1])
        return np.sign(x) * np.abs(x) ** power
    elif kind in ['s2', 's3', 's4']:
        power = int(kind[-1])
        return np.sign(x) * np.abs(x) ** (1. / power)
    elif kind in ['pp2', 'pp3', 'pp4']:
        t = params[0] * x + params[1]
        power = int(kind[-1])
        return np.sign(t) * np.abs(t) ** power
    elif kind in ['sp2', 'sp3', 'sp4']:
        t = params[0] * x + params[1]
        power = int(kind[-1])
        return np.sign(t) * np.abs(t) ** (1. / power)
        

    
def finv(y, kind='linear', params=None, eps=1e-9):
    if kind == 'linear':
        return y
    elif kind == 'g1':
        coef_ = np.sign(params[1]) * eps if np.abs(params[1]) < eps else params[1]
        return np.sign(y) * np.log(1 + np.abs(y) * np.exp(-params[0])) / coef_
    elif kind == 'g2':
        return np.sign(y) * (np.exp(-params[0]) * y) ** 2
    elif kind == 'g3':
        coef_ = np.sign(params[1]) * eps if np.abs(params[1]) < eps else params[1]
        return np.sign(y) * (np.log(1 + np.abs(y) * 
                                    np.exp(-params[0])) / coef_) ** 2
    elif kind == 'g4':
        coef_ = np.sign(params[1]) * eps if np.abs(params[1]) < eps else params[1]
        return np.sign(y) * (np.log(1 + np.abs(y) * 
                                    np.exp(-params[0])) / coef_) ** 3
    elif kind == 'g5':
        coef_ = np.sign(params[0]) * eps if np.abs(params[0]) < eps else params[0]
        return np.sign(y) * np.log(1 + np.abs(y / coef_))
    elif kind in ['p2', 'p3', 'p4']:
        return np.sign(y) * np.abs(y) ** (1. / int(kind[-1]))
    elif kind in ['s2', 's3', 's4']:
        return np.sign(y) * np.abs(y) ** int(kind[-1])
    elif kind in ['pp2', 'pp3', 'pp4']:
        t = params[0] * y + params[1]
        power = int(kind[-1])
        return np.sign(t) * np.abs(t) ** (1. / power)
    elif kind in ['sp2', 'sp3', 'sp4']:
        t = params[0] * y + params[1]
        power = int(kind[-1])
        return np.sign(t) * np.abs(t) ** power
    
    
def fgrad(x, kind='linear', params=None, eps=1e-9):
    if kind in NONPARAMETRICAL_TRANSROMATIONS:
        return None
    elif kind == 'g1':
        grad = np.stack([np.zeros_like(x), np.zeros_like(x)], axis=2)
        grad[:, :, 0] = np.sign(x) * np.exp(params[0]) * (np.exp(params[1] * np.abs(x)) - 1)
        grad[:, :, 1] = x * np.exp(params[0]) * (np.exp(params[1] * np.abs(x)) - 1)
        return grad
    elif kind == 'g2':
        grad = np.zeros_like(x)
        grad = np.sign(x) * np.exp(params[0]) * np.sqrt(np.abs(x))
        return grad
    elif kind == 'g3':
        grad = np.stack([np.zeros_like(x), np.zeros_like(x)], axis=2)
        grad[:, :, 0] = np.sign(x) * np.exp(params[0]) * (np.exp(params[1] * np.sqrt(np.abs(x))) - 1)
        grad[:, :, 1] = np.exp(params[0]) * (np.exp(params[1] * 
                                                    np.sqrt(np.abs(x))) - 1) / (2. * np.sqrt(np.abs(x)) + eps)
        return grad
    elif kind == 'g4':
        grad = np.stack([np.zeros_like(x), np.zeros_like(x)], axis=2)
        grad[:, :, 0] = np.sign(x) * np.exp(params[0]) * (np.exp(params[1] * np.abs(x) ** (1./3)) - 1)
        grad[:, :, 1] = np.exp(params[0]) * (np.exp(params[1] * 
                                                    np.abs(x) ** (1./3)) - 1)  / (3. * (np.abs(x)) ** (2./3) + eps)
        return grad
    elif kind == 'g5':
        grad = np.zeros_like(x)
        grad = np.sign(x) * (np.exp(np.abs(x)) - 1)
        return grad
    elif kind in ['pp2', 'pp3', 'pp4']:
        power = int(kind[-1])
        t = params[0] * x + params[1]
        grad = np.stack([np.zeros_like(x), np.zeros_like(x)], axis=2)
        grad[:, :, 0] = np.sign(t) * power * np.abs(t) ** (power - 1) * x
        grad[:, :, 1] = np.sign(t) * power * np.abs(t) ** (power - 1)
        return grad
    elif kind in ['sp2', 'sp3', 'sp4']:
        power = int(kind[-1])
        t = params[0] * x + params[1]
        grad = np.stack([np.zeros_like(x), np.zeros_like(x)], axis=2)
        grad[:, :, 0] = np.sign(t) * power * np.abs(t) ** (1. / power - 1) * x
        grad[:, :, 1] = np.sign(t) * power * np.abs(t) ** (1. / power - 1)
        return grad
        

def jacob(x, t, kind='linear', params=None, eps=1e-9):
    if kind in NONPARAMETRICAL_TRANSROMATIONS:
        return None
    elif kind in ['g1', 'g3', 'g4', 'pp2', 'pp3', 'pp4', 'sp2', 'sp3', 'sp4']:
        coef_ = 1. / np.sum(t ** 2)
        x_hat = f(x, kind=kind, params=params)
        grad = fgrad(x, kind=kind, params=params, eps=eps)
        a = coef_ * np.tensordot(np.tensordot(grad, x_hat.T, axes=([1], [0])), t, axes=([2], [0]))
        b = coef_ * np.tensordot(np.tensordot(x_hat, np.transpose(grad, axes=(1, 0, 2)), 
                                              axes=([1], [0])), t, axes=([1], [0]))
        return a[:, :, 0] + b[:, :, 0]
    elif kind in ['g2', 'g5']:
        coef_ = 1. / np.sum(t ** 2)
        x_hat = f(x, kind=kind, params=params)
        grad = fgrad(x, kind=kind, params=params, eps=eps)
        return coef_ * (grad.dot(x_hat.T).dot(t) + x_hat.dot(grad.T).dot(t))


def params_initialize(kind='linear'):
    if kind in NONPARAMETRICAL_TRANSROMATIONS:
        return None
    elif kind in ['g1', 'g3', 'g4', 'pp2', 'pp3', 'pp4', 'sp2', 'sp3', 'sp4']:
        return np.random.randn(2)
    elif kind in ['g2', 'g5']:
        return np.random.randn(1)