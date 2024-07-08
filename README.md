Description
===========
The contents of the repository are supplementary material to the article "Application of Portfolio Optimization to Achieve PersistentTime Series", DOI: https://doi.org/10.1007/s10957-024-02426-1. Optimal portfolio choice is used, aiming to achieve a portfolio that exhibits smoothness, strong positive auto-correlation, and long memory.


Repository structure
====================

- **data** directory: Contains intermediate results from the analysis that can be used instead of running the whole notebook from scratch, which would be quite time-consuming
    - ***price_df.csv***: daily adjusted close prices of [AAPL, NKE, GOOGL, AMZN] between *2005-01-01* and *2022-12-31*
    - ***results_df.csv***: results of running the portfolio optimization in each sub-sample of the prices

- **python** directory: contains the source code of the project
    - ***max_snr_portfolio_analysis.ipynb***: a Jupyter notebook that outlines how the analysis is done
    - ***optimal_weights.py***: implementation of the portfolio optimization techniques covered in the article
    - ***evaluate_weights.py***: implementation of fractal dimension estimator, Hurst exponent estimators, and a utility function for evaluating a given portfolio

- **pyproject.toml**: contains metadata for the content of the repository, like high-level dependencies.

- **poetry.lock**: the poetry (https://python-poetry.org/) lock file, containing the exact dependencies used while running the code.

Usage
=====

The notebook in the *Python* directory can be simply run. To make sure that your environment is identical, you should create a virtual environment via *poetry* (https://python-poetry.org/). The easiest way to install it is by running

`pip install poetry`

After installing *poetry*, navigate to the repository root and run

`poetry install`

to get a virtual environment identical to ours. If you only wish to run a *Jupyter* server from the new environment, simply run

`poetry run -- jupyter`
