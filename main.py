import numpy as np
from scipy.linalg import null_space
import itertools

def caratheodory(p, u):
    n = p.shape[0]
    d = p.shape[1]
    w = np.copy(u)
    indices = np.arange(n)
    while True:
        n = indices.shape[0]
        if n <= (d + 1):
            return p[indices], w, indices
        A = np.zeros((d, n-1))
        p0 = p[indices[0]]
        A = p[indices[1:]].T - np.tile(p0.reshape(-1, 1), (n-1,) )
        v = null_space(A)[:,0]
        v1 = - np.sum(v)
        v = np.insert(v, 0, v1)
        alpha = np.min([ (w[indices[i]] / v[i] ) for i in range(n) if v[i] > 0.0])
        w[indices] = w[indices] - (alpha * v)
        indices = np.where(w > 0)[0]

def flatten(l):
    return list(itertools.chain.from_iterable(l))

def fast_caratheodory(p, u, k):
    n = p.shape[0]
    d = p.shape[1]
    w = np.copy(u)
    indices = np.arange(n)
    while True:
        n = indices.shape[0]
        if n <= (d+1):
            return p[indices], w, indices
        if n < k:
            k = n
        int_split = np.insert(np.cumsum([(n + i) // k for i in range(k)]), 0, 0)
        clusters = []
        for i in range(k):
            l = int_split[i]
            r = int_split[i+1]
            clusters.append(list(range(l, r)))
        u_ = np.zeros(k)
        u_i = np.zeros((k, d))
        for i in range(k):
            c = clusters[i]
            u_[i] = np.sum(w[indices[c]])
            for ci in c:
                u_i[i] += w[indices[ci]] * p[indices[ci]]
            u_i[i] /= u_[i]
        u__, w__, indices_ = caratheodory(u_i, u_)
        for index in indices_:
            sum_u = np.sum(w[indices[clusters[index]]])
            for c in clusters[index]:
                w[indices[c]] = w__[index] * w[indices[c]] / sum_u
        tmp = []
        for index in indices_:
            tmp.append(clusters[index])
        tmp = flatten(tmp)
        indices = indices[tmp]
        for i in range(n):
            if i not in indices:
                w[i] = 0.0
        
def caratheodory_matrix(a):
    n = a.shape[0]
    d = a.shape[1]
    p = np.zeros((n, d*d))
    u = np.ones(n) / n
    for i in range(n):
        p[i] = (a[i].reshape(-1, 1) @ a[i].reshape(1, -1)).reshape(-1)
    c, w, indices = caratheodory(p, u)
    s = np.zeros((c.shape[0], d))
    for i, ai in enumerate(indices):
        s[i] = np.sqrt(n*w[ai]) * a[ai]
    return s 

def fast_caratheodory_matrix(a, k):
    n = a.shape[0]
    d = a.shape[1]
    p = np.zeros((n, d*d))
    u = np.ones(n) / n
    for i in range(n):
        p[i] = (a[i].reshape(-1, 1) @ a[i].reshape(1, -1)).reshape(-1)
    c, w, indices = fast_caratheodory(p, u, k)
    s = np.zeros((c.shape[0], d))
    for i, ai in enumerate(indices):
        s[i] = np.sqrt(n*w[ai]) * a[ai]
    return s 

if __name__ == '__main__':
    mat = np.random.random((1000, 3))
    print('Covariance matrix:')
    print(mat.T @ mat)

    print('Covariance matrix(caratheodory):')
    mat_ = caratheodory_matrix(mat)
    print(mat_.T @ mat_)

    print('Covariance matrix(fast caratheodory):')
    mat__ = fast_caratheodory_matrix(mat, 100)
    print(mat__.T @ mat__)