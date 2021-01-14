"""## Taken from scipy ticket
http://projects.scipy.org/scipy/attachment/ticket/648/designtools.py
"""


from numpy import asarray, array, arange, append, angle, complex128, float64, floor, roots
from numpy import nonzero, sign, mat, sin, cos, exp, zeros, log10, unique, fix, ceil
from numpy import ones, prod, pi, NaN, zeros_like, ravel, any, linspace, diff
from numpy import kron
from numpy.linalg import inv
from scipy.linalg import toeplitz, hankel

def firls(m, bands, desired, weight=None):
    """
    FIR filter design using least squares method.
    
    Inputs :
        m : oder of FIR filter

        bands : A montonic sequence containing the band edges.  All elements
                must be non-negative and less than 1 the sampling frequency
                as given in pi units.
        desired : A sequency with the same size of bands containing the desired gain
        in each of the specified bands
        weight : A relative weighting to give to each band region.
    
    Output :
        h  : coefficients of length m+1 fir filter.
    
    Example :
        h = firls(50, [0,0.2,0.3,1.0], [1,1,0,0],[1,5.0])
        
        Calculate impulse response for 51 tabs lowpass filter using least squares method
    with passband == [0,0.2*pi], stopband == [0.3*pi, pi],
        weight to passband == 1, weight to stopband == 5
    
    Note : This function is modified from signal package for octave 
    (http://octave.sourceforge.net/signal/index.html)
    """
    if weight==None : weight = ones(len(bands)//2)

    bands, desired, weight = array(bands), array(desired), array(weight)
    
    M = m//2;
    w = kron(weight, [-1,1])
    omega = bands * pi
    i1 = arange(1,M+1)
    
    # generate the matrix q
    # as illustrated in the above-cited reference, the matrix can be
    # expressed as the sum of a hankel and toeplitz matrix. a factor of
    # 1/2 has been dropped and the final filter hficients multiplied
    # by 2 to compensate.
    cos_ints = append(omega, sin(mat(arange(1,m+1)).T*mat(omega))).reshape((-1,omega.shape[0]))
    q = append(1, 1.0/arange(1.0,m+1)) * array(mat(cos_ints) * mat(w).T).T[0]
    q = toeplitz(q[:M+1]) + hankel(q[:M+1], q[M : ])
    
    # the vector b is derived from solving the integral:
    #
    #           _ w
    #          /   2
    #  b  =   /       w(w) d(w) cos(kw) dw
    #   k    /    w
    #       -      1
    #
    # since we assume that w(w) is constant over each band (if not, the
    # computation of q above would be considerably more complex), but
    # d(w) is allowed to be a linear function, in general the function
    # w(w) d(w) is linear. the computations below are derived from the
    # fact that:
    #     _
    #    /                          a              ax + b
    #   /   (ax + b) cos(nx) dx =  --- cos (nx) +  ------ sin(nx)
    #  /                             2                n
    # -                             n
    #

    
    enum = append(omega[::2]**2 - omega[1::2]**2, cos(mat(i1).T * mat(omega[1::2])) - cos(mat(i1).T * mat(omega[::2]))).flatten()
    deno = mat(append(2, i1)).T * mat(omega[1::2] - omega[::2])
    cos_ints2 = enum.reshape(deno.shape)/array(deno)
    
    d = zeros_like(desired)
    d[::2]  = -weight * desired[::2]
    d[1::2] =  weight * desired[1::2]
    
    b = append(1, 1.0/i1) * array(mat(kron (cos_ints2, [1, 1]) + cos_ints[:M+1,:]) * mat(d).T)[:,0]
    
    # having computed the components q and b of the  matrix equation,
    # solve for the filter hficients.
    a = (array(inv(q)*mat(b).T).T)[0]
    h = append( a[:0:-1], append(2*a[0],  a[1:]))
    return h