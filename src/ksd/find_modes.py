import tensorflow as tf
import tensorflow_probability as tfp

def pairwise_mahalanobis(inv_hessian, x):
    """Compute Mahalanobis distance
    inv_hessian: dim x dim
    x: dim
    """
    x_h = tf.expand_dims(x, axis=0) # 1 x dim
    res = x_h @ inv_hessian @ tf.transpose(x_h) # 1,
    return res[0, 0]

def merge_modes(inv_hessians: tf.Tensor, end_pts: tf.Tensor, threshold: float, log_prob: callable,
    threshold_ignore: float=1e-8):
    """Merge modes according to pairwise mahalanobis distance
    Args:
        end_pts: m x dim
        inv_hessians: m x dim x dim
        threshold_ignore: if prob of new mode < threshold_ignore, then do not
            keep it as a new mode.
    """
    M = end_pts.shape[0]
    mode_list = [end_pts[0, :]]
    inv_hessians_list = [inv_hessians[0, :, :]]
    
    for i in range(1, M):
        maha_dist_list = []
        end_pt_i = end_pts[i, :]
        inv_hess_i = inv_hessians[i]
        
        # compute pairwise Mahalanobis dist with the existing modes
        for j in range(len(mode_list)):
            inv_hess = 0.5 * (inv_hessians_list[j] + inv_hess_i)
            diff = mode_list[j] - end_pt_i
            maha_dist = pairwise_mahalanobis(inv_hess, diff)
            maha_dist_list.append(maha_dist)

        # find the mode with the closest distance
        argmin_i = tf.math.argmin(maha_dist_list).numpy()
        min_maha_dist = maha_dist_list[argmin_i]
        
        if min_maha_dist < threshold:
            # classify into closest mode
            closest_mode = mode_list[argmin_i]

            if log_prob(
                    tf.reshape(closest_mode, (1, -1))
                ) < log_prob(tf.reshape(end_pt_i, (1, -1))):
                # if current pt is better than local mode, swap
                mode_list[argmin_i] = end_pt_i
                inv_hessians_list[argmin_i] = inv_hess_i

        elif log_prob(tf.reshape(end_pt_i, (1, -1))) > tf.math.log(threshold_ignore):
            # store current pt as a new mode
            mode_list.append(end_pt_i)
            inv_hessians_list.append(inv_hess_i)
    
    # # remove first mode if its logprob is too low #TODO add back?
    # if not (log_prob(mode_list[0]) > tf.math.log(threshold_ignore)):
    #     mode_list.pop(0)
    #     inv_hessians_list.pop(0)

    return mode_list, inv_hessians_list

def run_bfgs(start_pts: tf.Tensor, log_prob_fn: callable, verbose: bool=False, grad_log: callable=None, **kwargs):
    """Run BFGS algorithm
    start_pts: M x dim
    """
    # define objective
    if not grad_log:
        def nll_and_grad(x):
            return tfp.math.value_and_gradient(
                lambda x: -log_prob_fn(x), # minus as we want to minimise
                x)
    else:
        def nll_and_grad(x):
            return (
                lambda x: -log_prob_fn(x), # minus as we want to minimise
                lambda x: -grad_log(x)
                )

    optim_results = tfp.optimizer.bfgs_minimize(
        nll_and_grad, initial_position=start_pts, parallel_iterations=1000, **kwargs,
    )

    if_converged = tf.experimental.numpy.all(optim_results.converged).numpy() # should return true if all converged

    if not if_converged and verbose:
        nstart_pts = start_pts.shape[0]
        not_conv = nstart_pts - tf.reduce_sum(tf.cast(optim_results.converged, dtype=tf.int32))
        Warning(f"{not_conv} of {nstart_pts} BFGS optim chains did not converge")

    return optim_results

def find_modes(start_pts, log_prob_fn, threshold, grad_log, threshold_ignore=1e-8, 
    max_iterations=50, **kwargs):
    """Run run_bfgs and merge_modes"""
    # run BFGS to find modes
    bfgs = run_bfgs(start_pts, log_prob_fn, grad_log, max_iterations=max_iterations)
    end_pts = bfgs.position
    inverse_hessian_estimate = bfgs.inverse_hessian_estimate

    # merge modes
    mode_list, inv_hess_list = merge_modes(inverse_hessian_estimate, end_pts, threshold, 
        log_prob_fn, threshold_ignore=threshold_ignore)

    return mode_list, inv_hess_list

def pairwise_directions(modes, return_index=False, ordered=True):
    """Compute v_{ij} = \mu_i - \mu_j for all 1 \leq i, j \leq len(modes).
    If ordered == True, then (i, j) and (j, i) are treated as two distinct pairs.
    Otherwise, only (i, j) where i < j are returned.

    Args:
        modes: list of mode vectors. Must have length >= 2
    """
    n = len(modes)
    dir_list = []
    index = []
    
    if ordered:
        for i in range(n):
            for j in range(n):
                if i != j:
                    dir = modes[i] - modes[j]
                    dir_list.append(dir)
                    index.append((i, j))
    else:
        for i in range(n-1):
            for j in range(i+1, n):
                dir = modes[i] - modes[j]
                dir_list.append(dir)
                index.append((i, j))
        
    if not return_index:
        return dir_list
    else:
        return dir_list, index
