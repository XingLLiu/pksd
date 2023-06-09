import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import trange
import argparse

from pksd.ksd import KSD, OSPKSD, SPKSD
from pksd.kernel import IMQ
from pksd.bootstrap import Bootstrap
import pksd.models as models
import pksd.models_np as models_np
import pksd.langevin as mcmc
from pksd.kgof.ksdagg import ksdagg_wild_test

import autograd.numpy as anp
import kgof
import kgof.density as kgof_density
import kgof.goftest as kgof_gof


def run_bootstrap_experiment(
    nrep,
    log_prob_fn,
    proposal,
    kernel,
    alpha,
    num_boot,
    T,
    jump_ls,
    n,
    MCMCKernel,
    method,
    rand_start=None,
    log_prob_fn_np=None,
    **kwargs,
):
    """
    Perform the following tests and repeat for nrep times:
        KSD, pKSD, spKSD, FSSD

    Args:
        method: One of "ksd", "ospksd", "spksd", "ksdagg", "fssd", or "all".
    """

    ntrain = n//2
    dim = proposal.event_shape[0]

    if method in ["ksd", "pksd", "ospksd", "spksd", "all"]:
        # initialise KSD instance
        ksd = KSD(target=target, kernel=kernel)

        # generate multinomial samples for bootstrap
        bootstrap = Bootstrap(ksd, n-ntrain)
        multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x ntest

        bootstrap_nopert = Bootstrap(ksd, n)
        multinom_samples_notrain = bootstrap_nopert.multinom.sample((nrep, num_boot)) # nrep x num_boot x n

    if method in ["fssd", "all"]:
        log_prob_fn_np_den = kgof_density.from_log_den(dim, log_prob_fn_np)

    # generate points for finding modes
    start_pts_all = tf.random.uniform(
        shape=(nrep, ntrain//2, dim), minval=-rand_start, maxval=rand_start,
    ) # nrep x ntrain x dim

    # initialise list to store results
    res = []

    iterator = trange(nrep)
    iterator.set_description(f"Running with sample size {n}")
    for iter in iterator:
        # set random seed
        tf.random.set_seed(iter + n)

        # generate samples
        sample_init = proposal.sample(n)

        # 1. KSD
        if method == "all" or method == "ksd":
            iterator.set_description("Running KSD")
            multinom_t = multinom_samples_notrain[iter, :] # nrep x num_boost x n

            # compute p-value
            ksd_rej, pval = bootstrap.test_once(
                alpha=alpha, num_boot=num_boot, X=sample_init, multinom_samples=multinom_t,
            )
            res.append(["KSD", ksd_rej, pval, iter])

        # 2. ospKSD
        if method == "all" or method == "ospksd":
            iterator.set_description("Running ospKSD")
    
            # train/test split
            sample_train, sample_test = sample_init[:ntrain], sample_init[ntrain:]
            
            # start optim from either randomly initialised points or training samples
            if rand_start:
                start_pts = tf.concat([sample_train[:(ntrain//4)], start_pts_all[iter, :(ntrain//4)]], axis=0) # ntrain x dim
            else:
                start_pts = sample_train # ntrain x dim

            # instantiate ospKSD class
            ospksd = OSPKSD(kernel=kernel, pert_kernel=MCMCKernel, log_prob=log_prob_fn)

            # find modes and Hessians
            ospksd.find_modes(start_pts, **kwargs)
                        
            # compute test statistic and p-value
            multinom_t = multinom_samples[iter, :] # nrep x num_boost x ntrain
            _, ospksd_pval = ospksd.test(
                xtrain=sample_train, 
                xtest=sample_test, 
                T=T,
                jump_ls=jump_ls, 
                num_boot=num_boot,
                multinom_samples=multinom_t,
            )
            ospksd_rej = float(ospksd_pval <= alpha)

            # store results
            res.append(["ospKSD", ospksd_rej, ospksd_pval, iter])

        # 3. spKSD
        if method == "all" or method == "spksd":
            iterator.set_description("Running spKSD")
        
            # start optim from either randomly initialised points or training samples
            start_pts = start_pts_all[iter] # n x dim

            # instantiate pKSD class
            pksd = SPKSD(kernel=kernel, pert_kernel=MCMCKernel, log_prob=log_prob_fn)

            # find modes and Hessians
            pksd.find_modes(start_pts, **kwargs)

            # compute test statistic and p-value 
            multinom_t = multinom_samples_notrain[iter, :] # nrep x num_boost x n
            _, pksd_pval = pksd.test(
                x=sample_init, 
                T=T,
                jump_ls=jump_ls, 
                num_boot=num_boot,
                multinom_samples=multinom_t, 
            )
            pksd_rej = float(pksd_pval <= alpha)

            # store results
            res.append(["spKSD", pksd_rej, pksd_pval, iter])

        ## 4. KSDAGG
        if method == "all" or method == "ksdagg":
            iterator.set_description("Running KSDAGG")
            x_t = sample_init
            ksdagg_rej = ksdagg_wild_test(
                seed=iter + n,
                X=x_t,
                log_prob_fn=log_prob_fn,
                alpha=alpha,
                beta_imq=0.5,
                kernel_type="imq",
                weights_type="uniform",
                l_minus=0,
                l_plus=10,
                B1=num_boot,
                B2=500, # num of samples to estimate level
                B3=50, # num of bisections to estimate quantile
            )
            res.append(["KSDAGG", ksdagg_rej, None, iter])

        ## 5. FSSD
        if method == "fssd" or method == "all" :
            dat = anp.array(sample_init, dtype=anp.float64)
            dat = kgof.data.Data(dat)
            tr, te = dat.split_tr_te(tr_proportion=0.2, seed=iter + n)

            # make sure to give tr (NOT te).
            # do the optimization with the options in opts.
            V_opt, gw_opt, _ = kgof_gof.GaussFSSD.optimize_auto_init(
                log_prob_fn_np_den,
                tr,
                J=10, # number of test locations (or features). Typically not larger than 10.
                reg=1e-2, # regularization parameter in the optimization objective
                max_iter=2000, # maximum number of gradient ascent iterations
                tol_fun=1e-7, # termination tolerance of the objective
            )

            fssd_opt = kgof_gof.GaussFSSD(log_prob_fn_np_den, gw_opt, V_opt, alpha)
            test_result = fssd_opt.perform_test(te)
            fssd_pval = test_result["pvalue"]
            rej = float(fssd_pval <= alpha)
            res.append(["FSSD", rej, fssd_pval, iter])

    res_df = pd.DataFrame(res, columns=["method", "rej", "pval", "seed"])

    return res_df

num_boot = 800 # number of bootstrap samples to compute critical val
alpha = 0.05 # test level
jump_ls = np.linspace(0.5, 1.5, 21).tolist() # std for discrete jump proposal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="", help="path to pre-saved results")
    parser.add_argument("--model", type=str, default="bimodal")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--mcmckernel", type=str, default="mh")
    parser.add_argument("--method", type=str, default="pksd")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000, help="sample size")
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--nrep", type=int, default=50)
    parser.add_argument("--ratio_t", type=float, default=0.5, help="max num of steps")
    parser.add_argument("--ratio_s", type=float, default=1.)
    parser.add_argument("--delta", type=float, default=8.)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--nmodes", type=int, default=10)
    parser.add_argument("--nbanana", type=int, default=2)
    parser.add_argument("--shift", type=float, default=0.)
    parser.add_argument("--dh", type=int, default=10, help="dim of h for RBM")
    parser.add_argument("--B_scale", type=float, default=6.)
    parser.add_argument("--noise_std", type=float, default=6.)
    parser.add_argument("--ratio_s_var", type=float, default=0.)
    parser.add_argument("--rand_start", type=float, default=None)
    parser.add_argument("--t_std", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=1.)
    args = parser.parse_args()
    model = args.model
    method = args.method
    seed = args.seed
    T = args.T
    n = args.n
    dim = args.dim
    nstart_pts = 20 * dim # num of starting points for finding modes
    nrep = args.nrep
    ratio_target = args.ratio_t
    ratio_sample = args.ratio_s
    k = args.k
    delta = args.delta # [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
    rand_start = args.rand_start
    mode_threshold = args.threshold # threshold for merging modes

    if args.mcmckernel == "mh":
        MCMCKernel = mcmc.RandomWalkMH
        mcmc_name = ""
    elif args.mcmckernel == "barker":
        MCMCKernel = mcmc.RandomWalkBarker
        mcmc_name = "barker"

    method = args.method + mcmc_name

    # create folders if not exist
    res_root = f"res/{model}"
    fig_root = f"figs/{model}"
    tf.io.gfile.makedirs(res_root)
    tf.io.gfile.makedirs(fig_root)

    # set random seed for all
    rdg = tf.random.Generator.from_seed(seed)
    
    # set model
    if model == "bimodal":
        model_name = f"{method}_steps{T}_ratio{ratio_target}_{ratio_sample}_k{k}_dim{dim}_seed{seed}_delta{delta}_n{n}{args.suffix}"
        create_target_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, return_logprob=True, ratio=ratio_target)
        create_sample_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, return_logprob=True, ratio=ratio_sample)

        # numpy version
        log_prob_fn_np = models_np.create_mixture_gaussian_kdim_logprobb(dim=dim, k=k, delta=delta, ratio=ratio_target, shift=0.)

    elif model == "t-banana":
        nmodes = args.nmodes
        nbanana = args.nbanana
        model_name = f"{method}_steps{T}_dim{dim}_nmodes{nmodes}_nbanana{nbanana}_ratiosvar{args.ratio_s_var}_t-std{args.t_std}_n{n}_seed{seed}"
        ratio_target = [1/nmodes] * nmodes
        
        random_weights = ratio_target + tf.exp(rdg.normal((nmodes,)) * args.ratio_s_var)
        ratio_sample = random_weights / tf.reduce_sum(random_weights)

        loc = rdg.uniform((nmodes, dim), minval=-tf.ones((dim,))*20, maxval=tf.ones((dim,))*20) # uniform in [-20, 20]^d

        b = 0.003
        create_target_model = models.create_mixture_t_banana(dim=dim, ratio=ratio_target, loc=loc, b=b,
            nbanana=nbanana, std=args.t_std, return_logprob=True)
        create_sample_model = models.create_mixture_t_banana(dim=dim, ratio=ratio_sample, loc=loc, b=b,
            nbanana=nbanana, std=args.t_std, return_logprob=True)

        # numpy version
        log_prob_fn_np = models_np.create_mixture_t_banana_logprob(dim=dim, ratio=ratio_target, loc=loc, 
            nbanana=nbanana, std=args.t_std, b=b)

    elif model == "gauss-scaled":
        model_name = f"{method}_steps{T}_seed{seed}"
        create_target_model = models.create_mixture_gaussian_scaled(ratio=ratio_target, return_logprob=True)
        create_sample_model = models.create_mixture_gaussian_scaled(ratio=ratio_sample, return_logprob=True)

    elif model == "rbm":
        dh = args.dh
        c_shift = args.shift
        B_scale = args.B_scale
        c_off = tf.concat([tf.ones(2) * c_shift, tf.zeros(dh-2)], axis=0)

        model_name = f"{method}_steps{T}_seed{seed}_dim{dim}_dh{dh}_shift{c_shift}_n{n}_B{B_scale}"
        create_target_model = models.create_rbm(B_scale=B_scale, c=0., dx=dim, dh=dh, burnin_number=2000, return_logprob=True)
        create_sample_model = models.create_rbm(B_scale=B_scale, c=c_off, dx=dim, dh=dh, burnin_number=2000, return_logprob=True)

        # numpy version
        log_prob_fn_np = models_np.create_rbm(B_scale=B_scale, c=0., dx=dim, dh=dh)

    elif model == "rbmStd":
        dh = args.dh
        noise_std = args.noise_std
        
        B_target = tf.cast(
            tf.random.normal([dim, dh]) > 0.,
            dtype=tf.float32,
        ) * 2. - 1
        B_sample = B_target + tf.random.normal([dim, dh]) * noise_std
        c = tf.random.normal((dh,))
        b = tf.random.normal((dim,)) * 0.1 # tf.zeros((dim,))

        model_name = f"{method}_steps{T}_seed{seed}_dim{dim}_dh{dh}_n{n}_noise{noise_std}"
        create_target_model = models.create_rbm_std(B=B_target, c=c, b=b, burnin_number=4000, return_logprob=True)
        create_sample_model = models.create_rbm_std(B=B_sample, c=c, b=b, burnin_number=4000, return_logprob=True)

        # numpy version
        log_prob_fn_np = models_np.create_rbm_std(B=B_target, c=c, b=b)

    elif model == "laplace":
        model_name = f"{method}_steps{T}_seed{seed}_dim{dim}_n{n}"
        normal_dist = tfd.MultivariateNormalDiag(tf.zeros(dim))
        
        create_target_model = models.create_laplace(dim=dim, return_logprob=True) 
        create_sample_model = normal_dist, normal_dist.log_prob
        
        # numpy version
        log_prob_fn_np = models_np.create_laplace()

    print(f"Running {model_name}")

    # target distribution
    target, log_prob_fn = create_target_model

    # proposal distribution
    proposal, log_prob_fn_proposal = create_sample_model
    
    # check if log_prob is correct
    models.check_log_prob(target, log_prob_fn)
    models.check_log_prob(proposal, log_prob_fn_proposal)

    # check if numpy version agrees with tf version
    if model != "t-banana":
        models_np.assert_equal_log_prob(target, log_prob_fn, log_prob_fn_np)

    # run experiment
    # with IMQ
    imq = IMQ(med_heuristic=True)

    res_df = run_bootstrap_experiment(
        nrep,
        log_prob_fn,
        proposal,
        imq,
        alpha,
        num_boot,
        T, 
        jump_ls,
        n=n,
        MCMCKernel=MCMCKernel,
        method=method,
        nstart_pts=nstart_pts, 
        log_prob_fn_np=log_prob_fn_np,
        threshold=mode_threshold,
        rand_start=rand_start,
    )

    # save res
    res_df.to_csv(f"{res_root}/{model_name}.csv", index=False)