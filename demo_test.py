    # QuantOpt: Institutional Portfolio Optimization
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    import quantopt as qo
    from quantopt.plotting import charts
    ## 1. Synthetic Data Generation
    n_assets = 10
    n_days = 1260 # 5 years
    # Base parameters
    mu_true = np.random.uniform(0.02 0.15 n_assets)
    vol_true = np.random.uniform(0.10 0.30 n_assets)
    # Generate a PSD correlation matrix
    factor_loadings = np.random.normal(0 1 (n_assets 3))
    cov_true = factor_loadings @ factor_loadings.T
    D = np.diag(1 / np.sqrt(np.diag(cov_true)))
    corr_true = D @ cov_true @ D
    cov_matrix = np.outer(vol_true vol_true) * corr_true
    # Simulate daily log returns
    dt = 1/252
    drift = (mu_true - 0.5 * vol_true**2) * dt
    L = np.linalg.cholesky(cov_matrix * dt)
    Z = np.random.standard_normal((n_days n_assets))
    log_ret = drift + Z @ L.T
    # Prices
    dates = pd.date_range(\2018-01-01\ periods=n_days freq=\B\)
    prices = pd.DataFrame(
    # Plot prices
    ## 2. Estimation (Expected Returns & Covariance)
    returns = qo.prices_to_returns(prices)
    # Estimate historical mean returns (with slight winsorization for robustness)
    mu_estimator = qo.MeanHistoricalReturn(returns)
    mu = mu_estimator.estimate()
    # Estimate covariance using a statistical PCA Factor Model (3 factors)
    cov_estimator = qo.FactorModelCovariance(returns n_components=3)
    Sigma = cov_estimator.estimate()
    ## 3. Optimization & The Efficient Frontier
    ef = qo.EfficientFrontier(mu Sigma l2_gamma=0.01)
    # 1. Max Sharpe
    w_sharpe = ef.max_sharpe()
    # 2. Min Volatility
    w_minvol = ef.min_volatility()
    # 3. Compute the full Efficient Frontier for plotting
    frontier_df = ef.efficient_frontier_points(n_points=40)
    # Plot
    benchmarks = {
    ## 4. Alternative Risk Frameworks: CVaR and Risk Parity
    # Equal Risk Contribution (Risk Parity)
    rp = qo.RiskParity(Sigma)
    w_rp = rp.optimize()
    # Conditional Value-at-Risk (CVaR) Optimization
    # Minimize 95% CVaR using historical scenarios
    cvar_opt = qo.CVaROptimizer(returns beta=0.95)
    w_cvar = cvar_opt.optimize()
    ## 5. Walk-Forward Backtesting
    # Define a backtest configuration with reasonable transaction costs (10bps spread)
    config = qo.BacktestConfig(
    # Define two strategies using the OptimizerFactory
    factory = qo.OptimizerFactory()
    strategies = {
    # Equal weight benchmark
    ew_returns = returns.mean(axis=1)
    ew_prices = (1 + ew_returns).cumprod() * 100
    # Run engine
    engine = qo.WalkForwardBacktester(
    results = engine.run_comparison(strategies config)
    # Plot Cumulative Returns
    rets_dict = {
    ## 6. Performance Tear Sheet
    rp_res = results[\Risk Parity\]
