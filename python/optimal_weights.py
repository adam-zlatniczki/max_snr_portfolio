import numpy as np
from scipy import linalg

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

from evaluate_weights import evaluate_weights


def random_portfolio(params):
#    import numpy as np
#    from evaluate_weights import evaluate_weights

    (seed, num_assets, cumul_returns_df, longonly) = params

    np.random.seed(seed)

    if longonly:
        # pick uniformly from the probability simplex
        x = np.zeros(num_assets+1)
        x[-1] = 1.0
        x[1:-1] = np.random.uniform(size=num_assets-1)
        weights = np.diff(sorted(x))
    else:
        # non-uniform sampling from the space of portfolios with short positions
        # uniform sampling is not feasible because this space is unbounded
        weights = np.random.uniform(low=-1.0, high=1.0, size=num_assets)
        weights = weights/np.sum(weights)

    H = evaluate_weights(weights, cumul_returns_df)

    return H

def min_var_portfolio(cov_matrix, longonly=False):
    num_assets = cov_matrix.shape[0]
    w = None

    if not longonly:
        w = np.dot(linalg.inv(cov_matrix), np.ones(cov_matrix.shape[0]))
        w /= np.sum(w)
    else:
        P = matrix(2*cov_matrix.values)
        q = matrix(np.zeros(num_assets))

        A = matrix(np.ones((1, num_assets)))
        b = matrix(1.0)

        G = matrix(-np.eye(num_assets))
        h = matrix(np.zeros(num_assets))

        sol = solvers.qp(P, q, G, h, A, b)
        if sol["status"] != "optimal":
            raise "Something went wrong, long-only Minimum Variance Portfolio couldn't be constructed!"
        w = sol["x"]

    return  w

def max_snr_portfolio(cov_matrix, exp_returns, longonly=False):
    w = None

    if not longonly:
        M = np.dot(exp_returns.values.reshape(-1,1), exp_returns.values.reshape(-1,1).T)
        evals, evects = linalg.eig(cov_matrix, M)
        w = evects[:, np.argmin(evals)]
        w /= np.sum(w)
    else:
        # calculate common parameters
        n = cov_matrix.shape[0]

        sigma_root = linalg.sqrtm(cov_matrix)

        c = np.zeros(n+1)
        c[0] = 1.0
        c = matrix(c)

        Gq = np.zeros((n+2, n+1))
        Gq[0, 0] = 1.0
        Gq[2:,1:] = sigma_root
        Gq = [matrix(-Gq)]

        hq = [matrix(np.zeros(n+2))]

        Gl = np.zeros((n+1, n+1))
        Gl[0, 1:] = -np.ones(n)
        Gl[1:, 1:] = -np.eye(n)
        Gl = matrix(Gl)

        hl = matrix(np.zeros(n+1))

        b = matrix([1.0])

        # check subspace of positive trends for a solution
        A = np.zeros((1, n+1))
        A[0, 1:] = exp_returns
        A = matrix(A)

        sol = solvers.socp(c, Gl=Gl, hl=hl, Gq=Gq, hq=hq, A=A, b=b)

        obj = None

        if sol["status"] == "optimal":
            # store current solution
            obj = sol["primal objective"]
            y = np.array(sol["x"])[1:]
            t = np.sum(y)

        # check subspace of negative trends for a solution
        A = np.zeros((1, n+1))
        A[0, 1:] = -exp_returns
        A = matrix(A)

        sol = solvers.socp(c, Gl=Gl, hl=hl, Gq=Gq, hq=hq, A=A, b=b)

        # compare solutions and keep the better one
        if sol["status"] == "optimal":
            if obj is None:
                # a solution was only available in the negative subspace, keep that one
                y = np.array(sol["x"])[1:]
                t = np.sum(y)
            else:
                # there's a solution in both subspaces, look for the better one
                if sol["primal objective"] < obj:
                    # the solution in the negative subspace is better, overwrite previous solution
                    y = np.array(sol["x"])[1:]
                    t = np.sum(y)

        w = y / t

    return w