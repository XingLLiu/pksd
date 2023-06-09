import autograd.numpy as anp
import kgof.density as kgof_density
# from scipy.special import logsumexp

def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.

        .. versionadded:: 0.11.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.

        .. versionadded:: 0.15.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.

        .. versionadded:: 0.12.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

        .. versionadded:: 0.16.0

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2

    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.

    Examples
    --------
    >>> from scipy.special import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107

    With weights

    >>> a = np.arange(10)
    >>> b = np.arange(10, 0, -1)
    >>> logsumexp(a, b=b)
    9.9170178533034665
    >>> np.log(np.sum(b*np.exp(a)))
    9.9170178533034647

    Returning a sign flag

    >>> logsumexp([1,2],b=[1,-1],return_sign=True)
    (1.5413248546129181, -1.0)

    Notice that `logsumexp` does not directly support masked arrays. To use it
    on a masked array, convert the mask into zero weights:

    >>> a = np.ma.array([np.log(2), 2, np.log(3)],
    ...                  mask=[False, True, False])
    >>> b = (~a.mask).astype(int)
    >>> logsumexp(a.data, b=b), np.log(5)
    1.6094379124341005, 1.6094379124341005

    """
    if b is not None:
        if anp.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -anp.inf

    a_max = anp.amax(a, axis=axis, keepdims=True)

    ## assert anp.sum(~anp.isfinite(a_max)) == 0.
    # if a_max.ndim > 0:
    #     a_max[~anp.isfinite(a_max)] = 0
    # elif not anp.isfinite(a_max):
    #     a_max = 0

    if b is not None:
        b = anp.asarray(b)
        tmp = b * anp.exp(a - a_max)
    else:
        tmp = anp.exp(a - a_max)

    # suppress warnings about log of zero
    with anp.errstate(divide="ignore"):
        s = anp.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = anp.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = anp.log(s)

    if not keepdims:
        a_max = anp.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out

def assert_equal_log_prob(dist, log_prob, log_prob_np):
  """Check if log_prob == dist.log_prob + const."""
  x = dist.sample(10)

#   res = anp.allclose(log_prob(x), log_prob_np(anp.array(x)), atol=1e-5)
  diff = log_prob(x) - log_prob_np(anp.array(x))
  res = anp.allclose(diff, diff[0], atol=5e-5)
  assert res, "log_prob and log_prob_np yield different values"

# end checker for log_prob_np implementation

def create_mixture_gaussian_kdim_logprobb(dim, k, delta, ratio=0.5, shift=0.):
    """
    Evaluate the log density at the points (rows) in X 
    of the standard isotropic Gaussian.
    Note that the density is NOT normalized. 
    
    X: n x d nd-array
    return a length-n array
    """
    a = [1. if x < k else 0. for x in range(dim)]
    a = anp.array(a)
    multiplier = delta / anp.sqrt(float(k))
    mean1 = anp.zeros(dim) + shift
    mean2 = multiplier * a + shift

    log_ratio1 = anp.log(ratio)
    log_ratio2 = anp.log(1-ratio)
    
    variance = 1

    def log_prob_fn(X):
      exp1 = -0.5 * anp.sum((X-mean1)**2, axis=-1) / variance + log_ratio1
      exp2 = -0.5 * anp.sum((X-mean2)**2, axis=-1) / variance + log_ratio2
      unden = anp.logaddexp(exp1, exp2) # n
      return unden

    return log_prob_fn


def multivariate_t_logprob(x, loc, Sigma_inv, df, dim):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    '''
    diff = x - loc
    prod = anp.matmul(diff, Sigma_inv) # n x d
    prod = anp.einsum("ij,ij->i", prod, diff) # n
    prod /= df
    log_den = -0.5 * (df + dim) * anp.log(1 + prod)
    return log_den


def banana_logprob(x, loc, b, df, dim, scale: float=10.):
    '''
    '''
    id_mat = anp.eye(dim)
    scale_mat = anp.concatenate([id_mat[:, :1] * scale, id_mat[:, 1:]], axis=-1)
    Sigma = anp.matmul(scale_mat, anp.transpose(scale_mat))
    Sigma_inv = anp.linalg.inv(Sigma)

    x_0 = x[..., 0:1]
    x_1 = x[..., 1:2] - b * x_0**2 + 100 * b
    # x[..., 1:2] = x[..., 1:2] - b * x_0**2 + 100 * b
    # log_den = multivariate_t_logprob(x, loc, Sigma_inv, df, dim)
    y = anp.concatenate([x_0, x_1, x[..., 2:]], axis=-1)
    log_den = multivariate_t_logprob(y, loc, Sigma_inv, df, dim)
    return log_den


def create_mixture_t_banana_logprob(dim, ratio, loc, nbanana, std, b):
  """
  """
  nmodes = len(ratio)
  assert nmodes >= nbanana, f"number of mixtures {nmodes} must be >= {nbanana}"
  
  t_scale = anp.sqrt(std * anp.sqrt(float(dim)) * anp.eye(dim))
  Sigma = t_scale @ anp.transpose(t_scale)
  Sigma_inv = anp.linalg.inv(Sigma)

  ratio = anp.array(ratio).reshape((-1, 1))
  # log_ratio = [anp.log(r) for r in ratio]

  def log_prob_fn(x):
    b_log_probs = [
      banana_logprob(
        x,
        loc=loc[i],
        b=b,
        df=7,
        dim=dim,
      ) for i in range(nbanana)
    ]
    # b_log_probs = [log_ratio[i] + b_log_probs[i] for i in range(nbanana)]

    t_log_probs = [
      multivariate_t_logprob(
        x, 
        loc=loc[nbanana+i], 
        Sigma_inv=Sigma_inv, 
        df=7, 
        dim=dim
      ) for i in range(nmodes-nbanana)
    ]
    # t_log_probs = [log_ratio[nbanana+i] + t_log_probs[i] for i in range(nmodes-nbanana)]

    log_probs = b_log_probs + t_log_probs

    log_prob = logsumexp(anp.stack(log_probs, axis=0), axis=0, b=ratio)
    return log_prob

  return log_prob_fn


def create_rbm(
  B_scale: anp.array=8.,
  c: anp.array=0.,
  dx: int=50,
  dh: int=40,
):
  """
  Generate data for the Gaussian-Bernoulli Restricted Boltzmann Machine (RBM) experiment.
  The entries of the matrix B are perturbed.
  This experiment was first proposed by Liu et al., 2016 (Section 6)

  Args:
    m: number of samples
    c: (dh,) either tf.Tensor or set to tf.zeros((dh,)) by default
    sigma: standard deviation of Gaussian noise
    dx: dimension of observed output variable
    dh: dimension of binary latent variable
    burnin_number: number of burn-in iterations for Gibbs sampler
  """
  # Model p
  B = anp.eye(anp.max((dx, dh)))[:dx, :dh] * B_scale
  b = anp.zeros(dx)
  c = c if isinstance(c, anp.ndarray) else anp.zeros(dh)

  dist = kgof_density.GaussBernRBM(B, b, c)
  dist.log_prob = dist.log_den

  return dist.log_prob


def create_rbm_std(
  B: anp.array=8.,
  c: anp.array=0.,
  b: anp.array=0.,
):
  """
  Generate data for the Gaussian-Bernoulli Restricted Boltzmann Machine (RBM) experiment.
  The entries of the matrix B are perturbed.
  This experiment was first proposed by Liu et al., 2016 (Section 6)

  Args:
    m: number of samples
    c: (dh,) either tf.Tensor or set to tf.zeros((dh,)) by default
    sigma: standard deviation of Gaussian noise
    burnin_number: number of burn-in iterations for Gibbs sampler
  """
  # Model p
  B = anp.array(B, dtype=anp.float64)
  b = anp.array(b, dtype=anp.float64)
  c = anp.array(c, dtype=anp.float64)

  dist = kgof_density.GaussBernRBM(B, b, c)
  dist.log_prob = dist.log_den

  return dist.log_prob


def create_laplace():
  """Create a Laplace ditributions with mean and var matched 
  to the standard Gaussian.
  """
  # loc = 0. and scale = 1/sqrt(2) match the 
  # moments of the standard gaussian
  scale = 1 / anp.sqrt(2.)

  def log_prob(x):
    lp = - anp.abs(x) / scale
    lp = anp.sum(lp, axis=-1)
    return lp

  return log_prob
