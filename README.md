Description
===========
The contents of the repository are supplementary material to an article on optimal portfolio choice, aiming to achieve a portfolo that exhibits smoothness, strong positive auto-correlation, and long memory.


Repository structure
====================

- **python** directory: contains the source code of the project
    - ***max_snr_portfolio_analysis.ipynb***: a Jupyter notebook that outlines how the analysis is done
    - ***optimal_weights.py***: implementation of the portfolio optimization techniques covered in the article
    - ***evaluate_weights.py***: implementation of fractal dimension estimator, Hurst exponent estimators, and a utility function for evaluating a given portfolio

- **data** directory: Contains intermediate results from the analysis that can be used instead of running the whole notebook from scratch, which would be quire time consuming
    - ***price_df.csv***: daily adjusted close prices of [AAPL, NKE, GOOGL, AMZN] between *2005-01-01* and *2022-12-31*
    - ***results_df.csv***: results of running the portfolio optimization in each subsample of the prices

- **venv** directory: contains the necessary files to create a conda virtual environment that is identical to the one used to run the notebook for the article (for easier reproducibility)
    - ***conda_env.yml***: use with *conda env create -f venv/conda_env.yml*