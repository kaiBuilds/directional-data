"""Functions for the Double Metropolis-Hastings algorithm for the Generalized von MIses.
"""

import argparse
from pathlib import Path

import numpy as np
from pydantic import PositiveInt
from tqdm import tqdm

from scipy.stats import ttest_ind_from_stats

from src import kernel


def get_rho(
    M: np.ndarray,
    ys: np.ndarray,
    ind_un: np.array,
    ind_obs: np.array,
    du: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the rho values for a given set of parameters.

    Args:
        M (numpy.ndarray): The matrix M.
        ys (numpy.ndarray): The ys values.
        ind_un (np.array): The indexes of the unobserved variable.
        ind_obs (np.array): The indexes of the observed variable.
        du (int): The dimension of the unobserved variable.

    Returns:
        rhoc (numpy.ndarray): The rhoc values.
        rhos (numpy.ndarray): The rhos values.
    """
    Muo = M[ind_un, :][:, ind_obs]

    if len(ys) > 0:
        rhoc = -np.dot(Muo, np.cos(ys))
        rhoc = np.reshape(rhoc, [-1, 1])

        rhos = -np.dot(Muo, np.sin(ys))
        rhos = np.reshape(rhos, [-1, 1])
    else:
        rhoc = np.zeros([du, 1], dtype=np.float64)
        rhos = np.zeros([du, 1], dtype=np.float64)

    return rhoc, rhos


def vmp_log_likelihood(varphi, M):
    """
    Compute the negative log-likelihood of the von Mises Process.
    Args:
        varphi (ndarray): The input angles.
        M (ndarray): The kernel matrix.
    Returns:
        float: The negative log-likelihood.
    """
    cos_terms = np.cos(varphi - varphi.T)
    return 0.5 * (cos_terms * M).sum()


def initialize_samples(N, every, d, ind_obs, ys):
    """
    Initializes the samples array for the Metropolis-Hastings algorithm.

    Args:
        N (int): The total number of samples.
        every (int): The interval at which to store samples.
        d (int): The dimensionality of the samples.
        ind_obs (int): The index of the observed values in the samples.
        ys (list): The observed values.

    Returns:
        samples (ndarray): The initialized samples array.
    """
    samples = np.zeros([d, int(N / every)])
    samples[ind_obs, :] = np.expand_dims(
        np.array(ys), 1
    )  # fix the observations for all the samples
    return samples


def initialize_observations(d, ind_obs, ys):
    varphi = np.zeros([d, 1])
    varphi[ind_obs, 0] = ys
    return varphi


def set_rv_initial_values(
    initial_sample: np.ndarray | None,
    du: int,
    ind_un: np.array,
) -> np.ndarray:
    if initial_sample is None or len(initial_sample) != du:
        thetas = np.random.rand(du) - 0.5
        # thetas = np.reshape(thetas, [du, 1])
    else:
        thetas = np.reshape(initial_sample[ind_un], [du, 1])
    return thetas


def get_dimensions(
    M: np.ndarray,
    ys: list[float] | np.ndarray,
    ind_obs: np.array,
) -> tuple[int, int, list[int]]:
    """
    Get the dimensions of the input matrix M.
    """
    d = M.shape[0]
    du = d - len(ys)  # number of unobserved variables in the model
    ind_un = ind_obs == False
    return d, du, ind_un


def get_decoupled_von_mises(
    A: np.ndarray,
    rhoc: np.ndarray,
    rhos: np.ndarray,
    theta: np.ndarray,
    du: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the decoupled von Mises distributions using the Hubbard-Stratonovich augmentation.

    Args:
        A (np.ndarray): The input array.
        rhoc (np.ndarray): The rhoc array.
        rhos (np.ndarray): The rhos array.
        theta (np.ndarray): The theta array.
        du (int): The value of du.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the mu and kappa arrays.
    """

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    z1 = A.dot(cos_theta) + np.random.normal(size=[du, 1])
    z2 = A.dot(sin_theta) + np.random.normal(size=[du, 1])

    kc = rhoc + (A.T).dot(z1)
    ks = rhos + (A.T).dot(z2)

    mu = np.arctan2(ks, kc)
    kappa = np.sqrt(kc**2 + ks**2)
    return mu, kappa


def gibbs_sampler_vmp(
    ys: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
    ind_obs: np.ndarray,
    ind_un: np.ndarray,
    N: PositiveInt,
    every: PositiveInt,
    verbose: bool = False,
    discard: PositiveInt = 20,
    initial_sample: np.ndarray | None = None,
    get_likelihoods: bool = False,
):
    """
    Performs Gibbs Sampling for the von Mises Process.

    Args:
        ys (np.ndarray): The observation ys array.
        M (np.ndarray): The M matrix.
        A (np.ndarray): The A matrix.
        ind_obs (np.ndarray): The observed indices array.
        un_obs (np.ndarray): The unobserved indices array.
        N (PositiveInt): The number of iterations.
        every (PositiveInt): The interval at which to store samples.
        verbose (bool, optional): Whether to display progress. Defaults to False.
        discard (PositiveInt, optional): The number of initial samples to discard. Defaults to 20.
        initial_sample (np.ndarray | None, optional): The initial sample. Defaults to None.
        get_likelihoods (bool, optional): Whether to compute negative log-likelihoods. Defaults to False.

    Returns:
        tuple: A tuple containing the samples and negative log-likelihoods.
    """
    d, du, ind_un = get_dimensions(M, ys, ind_obs)
    rhoc, rhos = get_rho(M, ys, ind_un, ind_obs, du)

    # initialize sampler variables
    theta = set_rv_initial_values(initial_sample, du, ind_un)
    samples = initialize_samples(N, every, d, ind_obs, ys)
    varphi = initialize_observations(d, ind_obs, ys)

    negative_log_likelihoods = np.zeros(int(N + 1 / every))

    # Gibbs sampler
    for i in range(-discard, N + 1):
        if verbose:
            print(f"iteration {i} / {N + 1} - {(i / N+1) * 100.}%")

        mu, kappa = get_decoupled_von_mises(
            A=A,
            rhoc=rhoc,
            rhos=rhos,
            theta=theta,
            du=du,
        )

        theta = np.random.vonmises(mu, kappa)  # theta in [-pi, pi]
        varphi[ind_un] = theta

        if i > 0 and i % every == 0:
            samples[ind_un, int(i / every) - 1] = theta[:, 0]
            if get_likelihoods:
                negative_log_likelihoods[int(i / every)] = vmp_log_likelihood(varphi, M)

    return samples, negative_log_likelihoods


def bridging_log_factor(
    theta: np.ndarray,
    Mp: np.ndarray,
    Ap: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
    bridging_steps: int,
    beta: np.ndarray | None = None,
    sqrt_beta: np.ndarray | None = None,
    sqrt_alpha: np.ndarray | None = None,
) -> float:
    """
    Calculate the bridging log factor.

    Args:
        theta (np.ndarray): The input array of shape (n, 1) representing the angles.
        Mp (np.ndarray): The input array of shape (n, n) representing the matrix Mp.
        Ap (np.ndarray): The input array of shape (n, n) representing the matrix Ap.
        M (np.ndarray): The input array of shape (n, n) representing the matrix M.
        A (np.ndarray): The input array of shape (n, n) representing the matrix A.
        bridging_steps (int): The number of bridging steps.
        beta (np.ndarray | None, optional): The input array of shape (bridging_steps + 1,) representing the beta values. Defaults to None.
        sqrt_beta (np.ndarray | None, optional): The input array of shape (bridging_steps + 1,) representing the square root of beta values. Defaults to None.
        sqrt_alpha (np.ndarray | None, optional): The input array of shape (bridging_steps + 1,) representing the square root of alpha values. Defaults to None.

    Returns:
        float: The bridging log factor.

    """
    d = M.shape[0]
    if beta is None:
        beta, sqrt_beta, sqrt_alpha = get_bridging_sequence_params(bridging_steps)

    k = 0
    Mk = beta[k] * M + (1 - beta[k]) * Mp
    Mkp = beta[k + 1] * M + (1 - beta[k + 1]) * Mp
    cosp = np.cos(theta - theta.T)

    bridge = 0.5 * (Mk * cosp).sum() - 0.5 * (Mkp * cosp).sum()

    for k in range(1, bridging_steps + 1):  # k samples
        cs = np.cos(theta)
        ss = np.sin(theta)

        y1 = sqrt_beta[k] * A.dot(cs) + np.random.normal(size=[d, 1])
        y2 = sqrt_beta[k] * A.dot(ss) + np.random.normal(size=[d, 1])
        y1p = sqrt_alpha[k] * Ap.dot(cs) + np.random.normal(size=[d, 1])
        y2p = sqrt_alpha[k] * Ap.dot(ss) + np.random.normal(size=[d, 1])

        kc = sqrt_beta[k] * (A.T).dot(y1) + sqrt_alpha[k] * (Ap.T).dot(y1p)
        ks = sqrt_beta[k] * (A.T).dot(y2) + sqrt_alpha[k] * (Ap.T).dot(y2p)

        kappa = np.sqrt(kc**2 + ks**2)
        mu = np.arctan2(ks, kc)
        theta = np.random.vonmises(mu, kappa)  # theta in [-pi, pi]

        Mk = Mkp
        Mkp = beta[k + 1] * M + (1 - beta[k + 1]) * Mp
        cosp = np.cos(theta - theta.T)

        bridge += 0.5 * (Mk * cosp).sum() - 0.5 * (Mkp * cosp).sum()

    return bridge


def get_bridging_sequence_params(
    bridging_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the bridging sequence parameters.

    Args:
        bridging_steps (int): The number of bridging steps.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three NumPy arrays:
            - beta: An array of bridging steps from 0 to 1.
            - sqrt_beta: The square root of beta.
            - sqrt_alpha: The square root of (1 - beta).
    """
    beta = np.arange(0, bridging_steps + 2) / (bridging_steps + 1)
    sqrt_beta = np.sqrt(beta)
    sqrt_alpha = np.sqrt(1 - beta)
    return beta, sqrt_beta, sqrt_alpha


def metropolis_hastings_factor(
    bridging_steps: PositiveInt,
    precision_matrix_m: np.ndarray,
    upper_cholesky_factor: np.ndarray,
    Mp: np.ndarray,
    Ap: np.ndarray,
    thetap: np.ndarray,
    theta: np.ndarray,
    mh_priors: np.ndarray,
    beta: np.ndarray | None = None,
    sqrt_beta: np.ndarray | None = None,
    sqrt_alpha: np.ndarray | None = None,
) -> float:
    """
    Calculates the Metropolis-Hastings factor for a given set of parameters.

    Args:
        bridging_steps (PositiveInt): The number of bridging steps.
        precision_matrix_m (np.ndarray): The precision matrix M.
        upper_cholesky_factor (np.ndarray): The upper Cholesky factor A.
        Mp (np.ndarray): The matrix Mp.
        Ap (np.ndarray): The matrix Ap.
        thetap (np.ndarray): The thetap array.
        theta (np.ndarray): The theta array.
        mh_priors (np.ndarray): The mh_priors array.
        beta (np.ndarray | None, optional): The beta array. Defaults to None.
        sqrt_beta (np.ndarray | None, optional): The sqrt_beta array. Defaults to None.
        sqrt_alpha (np.ndarray | None, optional): The sqrt_alpha array. Defaults to None.

    Returns:
        float: The calculated Metropolis-Hastings factor.
    """
    cos_theta_diff = np.cos(theta - theta.T)
    f1 = 0.5 * (precision_matrix_m * cos_theta_diff).sum()
    f3 = 0.5 * (Mp * cos_theta_diff).sum()
    if bridging_steps > 0:
        bridge = bridging_log_factor(
            theta=thetap,
            Mp=Mp,
            Ap=Ap,
            M=precision_matrix_m,
            A=upper_cholesky_factor,
            bridging_steps=bridging_steps,
            beta=beta,
            sqrt_beta=sqrt_beta,
            sqrt_alpha=sqrt_alpha,
        )
        mh = mh_priors - f3 + f1 + bridge
    else:
        cos_thetap_diff = np.cos(thetap - thetap.T)
        f2 = 0.5 * (precision_matrix_m * cos_thetap_diff).sum()
        f4 = 0.5 * (Mp * cos_thetap_diff).sum()
        mh = mh_priors - f3 - f2 + f4 + f1

    return mh


def geweke_test(l1s, l2s):
    """
    Perform the Geweke test for the given samples.
    The Geweke test is used to assess the convergence of Markov chain Monte Carlo (MCMC) samples.
    It compares the means of the first 10% and second half of the samples to test for any significant differences.
    The test is performed using the two-sample t-test based on sample statistics.

    Parameters:
    l1s (array-like): First set of samples.
    l2s (array-like): Second set of samples.

    Returns:
    tuple: A tuple containing the p-values of the Geweke test for l1s and l2s.
    """
    l1_samples = len(l1s)
    l1_first_sample = (l1s[: int(0.1 * l1_samples)],)
    l1_second_sample = l1s[int(0.5 * l1_samples) : l1_samples]

    l2_samples = len(l2s)
    l2_first_sample = l2s[: int(0.1 * l2_samples)]
    l2_second_sample = l2s[int(0.5 * l2_samples) : l2_samples]

    l1_geweke_pval = ttest_ind_from_stats(
        mean1=np.mean(l1_first_sample),
        std1=np.std(l1_first_sample),
        nobs1=len(l1_first_sample),
        mean2=np.mean(l1_second_sample),
        std2=np.std(l1_second_sample),
        nobs2=len(l1_second_sample),
        alternative="two-sided",
    )[1]

    l2_geweke_pval = ttest_ind_from_stats(
        mean1=np.mean(l2_first_sample),
        std1=np.std(l2_first_sample),
        nobs1=len(l2_first_sample),
        mean2=np.mean(l2_second_sample),
        std2=np.std(l2_second_sample),
        nobs2=len(l2_second_sample),
        alternative="two-sided",
    )[1]

    return l1_geweke_pval, l2_geweke_pval


def double_mh(
    ys: np.ndarray,
    xs: np.ndarray,
    ind_obs: np.array,
    N_thetap: int = 400,
    N_theta: int = 40,
    K: int = 1,
    Nsamples: int = 50_000,
    diagnostic: int = 1_000,
    discard: int = 20,
    l1: float = 0.01,
    l2: float = 0.01,
    sigma2_l1: float = 0.1,
    sigma2_l2: float = 0.1,
    seed: int = 42,
    save_path: str | Path = None,
) -> np.ndarray:

    geweke_pval_l1 = []
    geweke_pval_l2 = []
    acceptance_rate = []
    diagnostic_values = []

    # hyper_prior paramters
    stdev_l1 = np.sqrt(sigma2_l1)
    stdev_l2 = np.sqrt(sigma2_l2)

    ind_un = ind_obs == False

    X = kernel.get_squared_diffences(xs)
    M = kernel.get_se_kernel(X=X, l1=l1, l2=l2)
    A = kernel.get_matrix_a(M=M, ind_un=ind_un)
    d = X.shape[0]
    du = d - len(ys)  # number of unobserved variables in the model

    # Containers for the samples
    samples = np.zeros([d, Nsamples])
    l2s = np.zeros(Nsamples)
    l1s = np.zeros(Nsamples)

    # Set initial theta
    if len(ys) == 0:
        ys = np.array([])
        theta = set_rv_initial_values(initial_sample=ys, du=du, ind_un=ind_un)
    else:
        theta = set_rv_initial_values(initial_sample=None, du=du, ind_un=ind_un)

    i = 0
    acc = 0  # counter to track acceptance rate
    all_acc = 0

    np.random.seed(seed)  # reset seed for reproducibility at each sampling experiment
    for i in tqdm(range(i, Nsamples)):

        # propose new parameters
        l1p = np.random.normal(loc=l1, scale=stdev_l1)
        l2p = np.random.normal(loc=l2, scale=stdev_l2)
        Mp = kernel.get_se_kernel(X=X, l1=l1p, l2=l2p)
        Ap = kernel.get_matrix_a(M=Mp, ind_un=ind_un)

        # generate an approximately exact imaginary sample
        thetap, _ = gibbs_sampler_vmp(
            ys=np.array([]),
            M=Mp,
            A=Ap,
            ind_obs=ind_obs,
            ind_un=ind_un,
            N=N_thetap,
            every=N_thetap,
            verbose=False,
            discard=discard,
            initial_sample=theta,
        )  # [d,1], [d,d]

        mh_priors = (
            0.5 * (l1**2 - l1p**2) / sigma2_l1 + 0.5 * (l2**2 - l2p**2) / sigma2_l2
        )

        mh = metropolis_hastings_factor(
            bridging_steps=K,
            precision_matrix_m=M,
            upper_cholesky_factor=A,
            Mp=Mp,
            Ap=Ap,
            thetap=thetap,
            theta=theta,
            mh_priors=mh_priors,
        )

        if mh > 0 or np.random.random() < np.exp(mh):  # accept
            l1 = l1p
            l2 = l2p
            acc += 1
            all_acc += 1
            M = Mp
            A = Ap

        theta, _ = gibbs_sampler_vmp(
            ys=ys,
            M=M,
            A=A,
            ind_obs=ind_obs,
            ind_un=ind_un,
            N=N_theta,
            every=N_theta,
            verbose=False,
            discard=discard,
            initial_sample=theta,
        )  # [d,1]

        l2s[i] = l2
        l1s[i] = l1
        samples[:, i] = theta[:, 0]

        if i > 0 and i % diagnostic == 0:
            print("accept:", acc / diagnostic)
            acceptance_rate.append(acc / diagnostic)
            diagnostic_values.append(i)
            geweke_l1, geweke_l2 = geweke_test(l1s[:i], l2s[:i])
            geweke_pval_l1.append(geweke_l1)
            geweke_pval_l2.append(geweke_l2)
            print(f"Geweke's P-values L1: {geweke_l1:1.4e}, L2:{geweke_l2:1.4e}")

            acc = 0

    if save_path is not None:
        np.savez(
            save_path,
            samples=samples,
            l2s=l2s,
            l1s=l1s,
        )

    return samples, acceptance_rate, diagnostic_values, geweke_pval_l1, geweke_pval_l2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A double Metropolis-Hastings sampler using bridging distributions."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose mode.",
    )
    parser.add_argument(
        "--discard",
        type=int,
        default=50,
        help="Number of samples to discard for burning in. Default is 50.",
    )
    parser.add_argument(
        "--bridging",
        type=int,
        default=100,
        help="Number of bridging steps (K in the paper). Default is 100.",
    )
    parser.add_argument(
        "--number-of-samples",
        type=int,
        default=1_000_000,
        help="Number of samples to generate (N in the paper). Default is 1_000_000.",
    )
    parser.add_argument(
        "--intermediary-steps",
        type=int,
        default=400,
        help="Number of intermediary samples to obtain an 'imaginary sample' (N_theta in the paper). Default is 400.",
    )
    parser.add_argument(
        "--latent-steps",
        type=int,
        default=40,
        help="Number of samples for latent variables (N_thetap in the paper). Default is 40.",
    )
    parser.add_argument(
        "--l1",
        type=float,
        default=0.01,
        help="Mean for l1 prior. Default is 0.01.",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.01,
        help="Mean for l2 prior. Default is 0.01.",
    )
    parser.add_argument(
        "--sigma2-l1",
        type=float,
        default=0.1,
        help="Variance for l1 prior. Default is 0.1.",
    )
    parser.add_argument(
        "--sigma2-l2",
        type=float,
        default=0.1,
        help="Variance for l2 prior. Default is 0.1.",
    )
    parser.add_argument(
        "--diagnostic",
        type=int,
        default=1000,
        help="Interval for computing sampling diagnostics. Default is 1000.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Path to save the results to a file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Path to the input data. Must contain entries for 'xs', 'ys', and 'ind_obs'.",
    )
    args = parser.parse_args()

    if args.data_path:
        data = np.load(args.input_data_path)
    else:
        print("creating an example sampling from the prior.")
        NUMBER_OF_VARIABLES = 5
        NUMBER_OF_OBSERVED = 2

        data = dict(
            xs=np.linspace(-2, 2, NUMBER_OF_VARIABLES),
            ys=[],
            ind_obs=np.array([False for _ in range(NUMBER_OF_VARIABLES)]),
        )

    samples = double_mh(
        ys=data["ys"],
        xs=data["xs"],
        ind_obs=data["ind_obs"],
        N_thetap=args.intermediary_steps,
        N_theta=args.latent_steps,
        K=args.bridging,
        Nsamples=args.number_of_samples,
        diagnostic=args.diagnostic,
        discard=args.discard,
        l1=args.l1,
        l2=args.l2,
        sigma2_l1=args.sigma2_l1,
        sigma2_l2=args.sigma2_l2,
        seed=args.seed,
        save_path=args.save_path,
    )
