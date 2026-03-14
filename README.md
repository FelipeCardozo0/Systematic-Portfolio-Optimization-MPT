# QuantOpt: Institutional Mean-Variance Portfolio Optimization

## Abstract

QuantOpt is a Python library implementing production-grade portfolio construction workflows grounded in modern portfolio theory. It provides a unified pipeline spanning four stages: return estimation (historical mean, CAPM, Black-Litterman), covariance estimation (sample, exponentially weighted, Ledoit-Wolf OAS, factor model), portfolio optimization (mean-variance efficient frontier, equal risk contribution, CVaR minimization), and walk-forward backtesting with transaction costs. The library is designed for quantitative practitioners who require mathematically rigorous, easily auditable implementations with analytical gradients and PSD-guaranteed covariance matrices. It distinguishes itself from off-the-shelf alternatives through tight integration of the estimation and optimization layers, explicit support for L2 weight regularization, and a walk-forward backtesting engine with mark-to-market weight drift and a configurable transaction cost model.

---

## Repository Structure

```
MPT portfolio optimization/
├── quantopt/                        # Main Python package
│   ├── __init__.py                  # Public API: 39 exported symbols
│   ├── returns/
│   │   ├── preprocessing.py         # Price ↔ returns conversion, winsorization, cross-sectional demeaning
│   │   └── estimators.py            # MeanHistoricalReturn, CAPMReturn, BlackLittermanReturn
│   ├── risk/
│   │   ├── covariance.py            # SampleCovariance, EWMCovariance, LedoitWolfCovariance, FactorModelCovariance
│   │   └── metrics.py               # Risk decomposition: MRC, CRC, PRC, DR, HHI, VaR, CVaR, risk_report
│   ├── optimization/
│   │   ├── base.py                  # BaseOptimizer: portfolio_performance, clean_weights
│   │   ├── efficient_frontier.py    # EfficientFrontier: max_sharpe, min_volatility, efficient_return/risk, frontier_points
│   │   ├── risk_parity.py           # RiskParity: ERC / generalized risk budgeting
│   │   ├── cvar_optimizer.py        # CVaROptimizer: Rockafellar-Uryasev (2000)
│   │   ├── constraints.py           # ConstraintSet: fluent constraint builder
│   │   └── factory.py               # OptimizerFactory: dispatch by strategy name
│   ├── backtest/
│   │   └── engine.py                # WalkForwardBacktester, BacktestConfig, TransactionCostModel, BacktestResult
│   ├── analytics/
│   │   └── performance.py           # Sharpe, Sortino, Calmar, Omega, drawdown, factor_attribution, rolling_metrics
│   ├── plotting/
│   │   └── charts.py                # Matplotlib/Seaborn visualization utilities
│   └── utils/
│       └── validation.py            # Input validation, PSD projection (eigenclip)
├── notebooks/
│   ├── demo.ipynb                   # Short usage demonstration
│   └── visualization.ipynb          # Full documentation notebook (this file generates docs/figures/)
├── tests/                           # pytest unit test suite (9 test modules)
├── docs/
│   └── figures/                     # Auto-generated figure directory (created by visualization.ipynb)
├── pyproject.toml                   # Build configuration (setuptools, Black, MyPy, pytest)
├── setup.py                         # Package metadata, version, dependencies
└── requirements.txt                 # Pinned runtime dependencies
```

---

## Installation

### Requirements

Python 3.11 or later is required. The table below lists runtime dependencies and their minimum compatible versions.

| Package       | Minimum Version | Purpose                                              |
|---------------|-----------------|------------------------------------------------------|
| numpy         | 1.26            | Numerical linear algebra, random number generation   |
| pandas        | 2.1             | Time series data frames, resampling, alignment       |
| scipy         | 1.11            | Numerical optimization (SLSQP), probability functions|
| scikit-learn  | 1.3             | OAS covariance estimator, PCA for factor model       |
| matplotlib    | 3.8             | Figure rendering, axes formatting                    |
| seaborn       | 0.13            | Statistical graphics, heatmaps                       |

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<username>/quantopt.git
cd quantopt

# 2. Create an isolated virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install in editable mode with development extras
pip install -e ".[dev]"

# 4. Verify installation
python -c "import quantopt; print(quantopt.__version__)"
# Expected output: 1.0.0

# 5. Run the test suite
pytest tests/ -v

# 6. Launch the documentation notebook
jupyter notebook notebooks/visualization.ipynb
```

---

## Quickstart

The following example generates synthetic prices, fits a Ledoit-Wolf covariance matrix, constructs Black-Litterman expected returns, and computes the maximum Sharpe ratio portfolio.

```python
import numpy as np
import pandas as pd
from quantopt.returns.estimators import BlackLittermanReturn
from quantopt.risk.covariance import LedoitWolfCovariance
from quantopt.optimization.efficient_frontier import EfficientFrontier

# ── Synthetic price data (GBM) ───────────────────────────────────────────────
rng     = np.random.default_rng(0)
T, N    = 504, 8
tickers = [f"ASSET_{chr(65+i)}" for i in range(N)]
dates   = pd.bdate_range("2022-01-03", periods=T)

log_ret = rng.normal(0.0004, 0.012, size=(T, N))
prices  = pd.DataFrame(
    100 * np.exp(np.cumsum(log_ret, axis=0)),
    index=dates, columns=tickers,
)
returns = pd.DataFrame(log_ret, index=dates, columns=tickers)

# ── Covariance estimation ─────────────────────────────────────────────────────
lw    = LedoitWolfCovariance().fit(returns)
Sigma = lw.covariance()
print(f"LW shrinkage alpha: {lw.shrinkage_:.4f}")

# ── Black-Litterman returns (equal-weight prior, no active views) ─────────────
market_caps = pd.Series(np.ones(N) / N, index=tickers)
bl = BlackLittermanReturn(
    market_caps=market_caps, risk_aversion=2.5, tau=0.05,
).fit(returns)
mu = bl.expected_returns()

# ── Mean-variance optimization ────────────────────────────────────────────────
ef = EfficientFrontier(mu=mu, Sigma=Sigma)
weights = ef.max_sharpe(risk_free_rate=0.02)

ret, vol, sr = ef.portfolio_performance(mu=mu, Sigma=Sigma, risk_free_rate=0.02)
print(f"Expected return : {ret:.2%}")
print(f"Volatility      : {vol:.2%}")
print(f"Sharpe ratio    : {sr:.3f}")
print("\nPortfolio weights:")
print(ef.clean_weights(threshold=0.005).round(4).to_string())
```

---

## Methods

### 5.1 Return Estimation

#### 5.1.1 Historical Mean (Simple and EWM)

The historical mean estimator computes the time-average of observed log returns and annualizes by geometric compounding: $\hat{\mu}_i = (1 + \bar{r}_i)^{252} - 1$, where $\bar{r}_i = T^{-1}\sum_t r_{i,t}$. The exponentially weighted variant assigns weight $w_t \propto \lambda^{T-t}$ with decay $\lambda = 1 - 2/(s+1)$ for span $s$, down-weighting stale observations. Key parameters: `frequency` (default 252), `exponential_weighting` (default `False`), `span` (default 60). The simple mean is the maximum-likelihood estimator under i.i.d. Gaussian returns but carries high sampling variance proportional to $\sigma_i/\sqrt{T}$. The EWM variant is preferred when drift is non-stationary.

#### 5.1.2 CAPM-Implied Returns

The CAPM estimator computes $\mathbb{E}[R_i] = R_f + \beta_i(\mathbb{E}[R_m] - R_f)$, where $\beta_i = \text{Cov}(R_i^{\text{exc}}, R_m^{\text{exc}})/\text{Var}(R_m^{\text{exc}})$ is estimated via OLS. Key parameters: `market_returns` (required, `pd.Series`), `risk_free_rate` (default 0.0). The CAPM estimator is preferred over the historical mean when idiosyncratic noise dominates the sample and a reliable market proxy is available.

#### 5.1.3 Black-Litterman Posterior

The Black-Litterman model blends a market-equilibrium prior $\boldsymbol{\Pi} = \delta\boldsymbol{\Sigma}\mathbf{w}_{\text{mkt}}$ with $K$ investor views encoded in the $K \times N$ pick matrix $\mathbf{P}$ and view vector $\mathbf{Q}$. The posterior follows the Master Formula: $\boldsymbol{\mu}_{\text{BL}} = \mathbf{M}[({\tau\boldsymbol{\Sigma}})^{-1}\boldsymbol{\Pi} + \mathbf{P}^\top\boldsymbol{\Omega}^{-1}\mathbf{Q}]$, $\mathbf{M}^{-1} = (\tau\boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top\boldsymbol{\Omega}^{-1}\mathbf{P}$. Key parameters: `market_caps` (required), `risk_aversion` (default 2.5), `tau` (default 0.05, recommended range 0.01–0.10). The Black-Litterman estimator is the preferred choice for production portfolios because it shrinks extreme historical estimates toward the equilibrium, reducing estimation error sensitivity.

---

### 5.2 Covariance Estimation

#### 5.2.1 Sample Covariance

The standard unbiased sample covariance $\mathbf{S} = (T-1)^{-1}\sum_t(\mathbf{r}_t - \bar{\mathbf{r}})(\mathbf{r}_t - \bar{\mathbf{r}})^\top$ is annualized by multiplying by the frequency. A PSD projection (eigenclip) is applied if any eigenvalue falls below $10^{-8}$. The condition number is computed and a warning is raised above 1000. This estimator is appropriate only when $T/N \gg 10$.

#### 5.2.2 Exponentially Weighted Covariance (RiskMetrics)

The EWM covariance assigns exponentially decaying weights $w_t \propto \lambda^{T-t}$ and computes the weighted outer-product sum: $\hat{\boldsymbol{\Sigma}} = \sum_t w_t (\mathbf{r}_t - \bar{\mathbf{r}}_w)(\mathbf{r}_t - \bar{\mathbf{r}}_w)^\top$. Key parameter: `span` (default 60, recommended range 20–120). The EWM covariance is preferred in regime-changing environments where recent volatility structure is more informative than the full historical window.

#### 5.2.3 Ledoit-Wolf Oracle Approximating Shrinkage

The OAS estimator (Chen et al., 2010) linearly blends the sample covariance with a multiple of the identity matrix: $\hat{\boldsymbol{\Sigma}}^{\text{LW}} = (1-\alpha)\mathbf{S} + \alpha\,\bar{\mu}_S\mathbf{I}$. The optimal $\alpha^*$ is computed analytically from the data under the OAS criterion, minimizing the estimated Frobenius-norm loss. Implementation delegates to `sklearn.covariance.OAS`. The shrinkage intensity $\alpha$ is stored in `lw.shrinkage_`. Ledoit-Wolf OAS is the default covariance estimator for the Max Sharpe backtest and is recommended whenever $N/T > 0.1$.

#### 5.2.4 Barra-Style Factor Model (PCA)

The PCA factor model extracts $K$ orthogonal latent factors via principal component analysis: $\boldsymbol{\Sigma} = \mathbf{B}\mathbf{F}\mathbf{B}^\top + \mathbf{D}$, where $\mathbf{B}$ ($N \times K$) are the factor loadings, $\mathbf{F}$ ($K \times K$) is the diagonal factor covariance, and $\mathbf{D} = \text{diag}(d_1,\ldots,d_N)$ is the idiosyncratic variance matrix. Key parameters: `n_factors` (fixed $K$, or `None` to select automatically by `variance_threshold`, default 0.80). The factor model is preferred over shrinkage when a block-correlation structure is known a priori and the idiosyncratic terms are plausibly uncorrelated.

---

### 5.3 Portfolio Optimization

#### 5.3.1 Mean-Variance Efficient Frontier

`EfficientFrontier` accepts a `pd.Series` of expected returns `mu` and a `pd.DataFrame` covariance matrix `Sigma`. All methods use SLSQP with analytical gradients.

**Maximum Sharpe Ratio.** Maximizes $({\mathbf{w}^\top\boldsymbol{\mu} - R_f})/{\sqrt{\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}}}$ subject to long-only and sum-to-one constraints. Five Dirichlet random restarts are used; the globally best objective value across all restarts is returned. The analytical gradient avoids finite-difference approximations. Preferred for total-return mandates.

**Minimum Variance Portfolio.** Minimizes $\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}$ with a single convex-quadratic restart from the equal-weight initial guess. The unique global optimum is guaranteed for PSD $\boldsymbol{\Sigma}$.

**Efficient Return / Efficient Risk.** `efficient_return(target_return)` adds a return inequality constraint $\mathbf{w}^\top\boldsymbol{\mu} \ge \mu^*$; `efficient_risk(target_volatility)` adds a variance inequality $\mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w} \le \sigma^{*2}$. These are used internally by `efficient_frontier_points()` to trace the full frontier.

**L2 Weight Regularization.** The penalty $\gamma\mathbf{w}^\top\mathbf{w}$ is added to all objectives, controlled by `l2_gamma` (default 0.0, recommended range 0–2). As $\gamma \to \infty$, the solution converges toward equal weight. See Section 9 of the notebook for the Sharpe-HHI tradeoff curve.

#### 5.3.2 Equal Risk Contribution (Risk Parity)

`RiskParity` minimizes $\sum_i(\text{CRC}_i - b_i\sigma_p)^2$ subject to $\sum_i w_i = 1$, $w_i \ge 0$, where $\text{CRC}_i = w_i(\boldsymbol{\Sigma}\mathbf{w})_i/\sigma_p$ and $b_i = 1/N$ for equal budgets. The implementation uses 15 Dirichlet random restarts and tolerance `ftol=1e-12`. Risk parity is preferred when the investor is agnostic about expected returns and prioritizes diversification of risk.

#### 5.3.3 CVaR Minimization (Rockafellar-Uryasev)

`CVaROptimizer` minimizes the auxiliary convex function $\min_\alpha \alpha + (1-\beta)^{-1}T^{-1}\sum_t\max(-\mathbf{r}_t^\top\mathbf{w} - \alpha, 0)$, which is jointly convex in $(\mathbf{w}, \alpha)$. The decision variable is the augmented vector $[\mathbf{w}^\top, \alpha]^\top \in \mathbb{R}^{N+1}$. The implementation computes analytical gradients with respect to both $\mathbf{w}$ and $\alpha$. Key parameters: `beta` (default 0.95), `weight_bounds` (default $(0,1)$), `mean_return_target` (optional floor constraint). CVaR optimization is preferred for tail-risk-sensitive mandates such as insurance-linked strategies and pension portfolios subject to VaR regulatory constraints.

---

### 5.4 Constraint Framework

The `ConstraintSet` class provides a fluent interface for composing optimization constraints:

- `.long_only()`: Restricts $w_i \in [0, 1]$ for all $i$.
- `.long_short(gross_exposure, net_exposure)`: Sets gross ($\sum|w_i| \le G$) and net ($|{\sum w_i}| \le N_e$) exposure bounds.
- `.max_position(limit)` / `.min_position(floor)`: Per-asset upper and lower bounds.
- `.sum_to_one()`: Equality constraint $\sum_i w_i = 1$.
- `.sector_neutral(sector_map, max_deviation)`: Limits deviation of sector weights from a specified benchmark.
- `.max_turnover(limit, current_weights)`: Enforces one-way turnover $\le$ `limit` via an inequality constraint.
- `.factor_exposure(loadings, min_exp, max_exp)`: Bounds portfolio factor exposures $\mathbf{B}^\top\mathbf{w}$.

The `.bounds()` and `.constraints()` methods return `scipy.optimize`-compatible tuples and dictionaries, respectively.

---

### 5.5 Walk-Forward Backtesting

`WalkForwardBacktester` accepts a price `pd.DataFrame`, an `optimizer_factory` callable (signature `Callable[[pd.DataFrame], BaseOptimizer]`), and a `BacktestConfig` dataclass. At each rebalancing date (monthly by default), the engine extracts the lookback window, invokes the factory to construct and return an unfitted optimizer, calls `optimizer.optimize()`, applies the turnover constraint if configured, and deducts transaction costs. Between rebalance dates, portfolio weights drift mark-to-market. The returned `BacktestResult` contains: `portfolio_returns`, `gross_returns`, `weights_history`, `realized_weights`, `turnover_history`, `transaction_costs`, `rebalance_dates`, `summary`, `cumulative_returns`, and `dollar_value`.

---

## Figures and Results

### Data Overview

![Figure 1](docs/figures/figure_01_price_series.png)

*Simulated price paths, correlation structure, return densities, and cross-sectional volatility ordering for the 10-asset GBM universe.*

The normalized price series illustrates the range of outcomes across the 10 simulated assets over the 5-year window. The correlation heatmap confirms that the three-factor generation process produces a realistic off-diagonal structure with cross-asset correlations in the range $[-0.2, 0.7]$. The KDE panel confirms that per-asset return distributions closely approximate the Gaussian density used in the simulation, with no heavy-tail artifacts visible at this scale. The volatility bar chart spans approximately 17% to 35% annualized, reflecting the range of $\sigma_i^{\text{daily}}$ parameters.

![Figure 2](docs/figures/figure_02_return_statistics.png)

*QQ plot of pooled daily log returns versus standard normal, and rolling 63-day cross-asset realized volatility.*

The QQ plot shows near-perfect alignment with the standard normal reference line in the central mass of the distribution, with slight deviations in the extreme tails. This is expected under GBM simulation; on real equity data, the tail deviations would be substantially larger. The rolling volatility panel shows the absence of volatility clustering by construction, which is a known limitation of the GBM framework.

---

### Return Estimators

![Figure 3](docs/figures/figure_03_return_estimators.png)

*Grouped bar chart comparing true drift parameters against historical, EWM, and CAPM estimates; scatter plot of Black-Litterman equilibrium prior versus posterior.*

The historical estimators track the true drift parameters within the expected sampling noise band. The CAPM estimator compresses returns toward the market mean, reflecting the beta-scaling of the single-factor model. The Black-Litterman scatter plot illustrates how the two encoded views pull the posterior away from the 45-degree line for ASSET_D, ASSET_C, and ASSET_B while the remaining assets remain close to the equilibrium prior.

---

### Covariance Estimation

![Figure 4](docs/figures/figure_04_covariance_comparison.png)

*Correlation matrix heatmaps for all four covariance estimators: sample, EWM, Ledoit-Wolf OAS, and PCA factor model.*

The sample and factor model correlation matrices tend to preserve the block structure more faithfully, while the Ledoit-Wolf estimator produces a smoother, slightly compressed off-diagonal pattern due to shrinkage toward the identity. The EWM estimator reflects the most recent volatility regime and may differ from the sample matrix in episodes of elevated recent comovement.

![Figure 5](docs/figures/figure_05_covariance_diagnostics.png)

*Eigenvalue spectra across estimators; Ledoit-Wolf shrinkage intensity; factor model systematic versus idiosyncratic variance decomposition.*

The eigenvalue spectrum panel shows that the sample covariance has the highest variance in eigenvalues, particularly for the largest and smallest factors. Ledoit-Wolf shrinkage compresses the spectrum toward the mean eigenvalue, improving condition number. The variance decomposition confirms that the three-factor structure generates moderate systematic fractions (30–70%) across assets, which is consistent with a realistic multi-factor equity universe.

---

### Portfolio Optimization

![Figure 6](docs/figures/figure_06_efficient_frontier.png)

*Efficient frontier colored by Sharpe ratio, with the Capital Market Line, minimum variance portfolio, and tangency portfolio highlighted.*

The efficient frontier illustrates the classical risk-return tradeoff. The tangency portfolio achieves the highest Sharpe ratio and lies at the point of tangency between the Capital Market Line and the frontier. Assets with the highest volatility lie far to the right; the optimizer correctly excludes or underweights these unless their expected return premium compensates. The CML extends beyond the tangency point to indicate the leveraged region.

![Figure 7](docs/figures/figure_07_optimal_weights.png)

*Portfolio weights for the maximum Sharpe, minimum variance, and efficient return (target set at 80% of the maximum Black-Litterman return) portfolios.*

The max Sharpe portfolio concentrates weight in assets with favorable return-to-risk ratios as estimated by the Black-Litterman model. The minimum variance portfolio favors low-volatility assets regardless of expected return. The efficient return portfolio is constrained to a target equal to 80% of the highest Black-Litterman posterior return, which is always feasible by construction; it sits between the minimum variance and tangency portfolios on the frontier.

---

### Risk Parity

![Figure 8](docs/figures/figure_08_risk_parity.png)

*Weight and risk contribution comparisons between the max Sharpe and risk parity portfolios; diversification ratio and HHI across strategies.*

The risk parity portfolio produces near-uniform percentage risk contributions (approximately 10% each), as required by the ERC condition. The max Sharpe portfolio, by contrast, concentrates most of its risk budget in the two or three highest-return assets. The diversification ratio is highest for risk parity, confirming its superior diversification of volatility sources.

---

### CVaR Optimization

![Figure 9](docs/figures/figure_09_cvar_optimization.png)

*Loss distribution with VaR and CVaR marks; CVaR-versus-beta curve; CVaR decomposition by asset; weight comparison.*

The loss distribution panel visually separates the tail region from the body of the loss distribution. CVaR lies strictly to the right of VaR, consistent with its definition as the expected loss in the tail. The CVaR-versus-beta curve shows the expected monotone relationship: both risk measures increase with confidence level, with CVaR increasing faster. The weight comparison reveals how the CVaR optimizer reallocates away from the highest-volatility assets relative to the max Sharpe solution.

---

### Walk-Forward Backtest

![Figure 10](docs/figures/figure_10_cumulative_returns.png)

*Net-of-costs cumulative portfolio value for all four strategies over the 2019–2024 simulation period.*

All four strategies generate positive terminal values over the synthetic GBM horizon, which is expected given the positive drift parameters. Relative rankings vary across subperiods, illustrating that no single strategy uniformly dominates. Any drawdown regions exceeding 10% (if present in the simulation run) are indicated by shaded grey bands.

![Figure 11](docs/figures/figure_11_drawdown.png)

*Drawdown time series for all strategies; monthly one-way turnover by strategy.*

The drawdown panel illustrates the maximum peak-to-trough decline for each strategy. Risk parity and minimum volatility portfolios typically exhibit shallower drawdowns due to their lower weight concentration. The turnover panel shows that CVaR optimization and max Sharpe portfolios rebalance more aggressively, incurring higher transaction costs than risk parity.

![Figure 12](docs/figures/figure_12_rolling_metrics.png)

*Rolling 63-day Sharpe ratio, realized volatility, and maximum drawdown for the best-performing strategy.*

The rolling metrics confirm that performance is not uniformly distributed over time. The rolling Sharpe ratio crosses zero in periods of market stress, and the rolling volatility shows the absence of GARCH-type clustering by construction of the GBM simulation. The global minimum drawdown point is annotated.

![Figure 13](docs/figures/figure_13_attribution.png)

*Grouped performance metrics across strategies; factor attribution (alpha and market beta) for the best strategy.*

The grouped bar chart summarizes the four key performance metrics in a single panel for easy cross-strategy comparison. The factor attribution regression quantifies how much of the best strategy's return is explained by the equal-weight market factor and how much constitutes genuine alpha.

---

### Risk Report

![Figure 14](docs/figures/figure_14_risk_dashboard.png)

*Six-panel risk dashboard: annualized return, volatility, Sharpe ratio, maximum drawdown, CVaR 95%, and effective number of positions for all strategies.*

The risk dashboard enables rapid multi-dimensional comparison of all four strategies. Strategies with high Sharpe ratios tend to have low effective N (concentrated positions), while risk parity achieves the highest diversification at the cost of a lower Sharpe ratio under GBM simulation.

---

### Sensitivity Analysis

![Figure 15](docs/figures/figure_15_sensitivity.png)

*Sharpe ratio and HHI as a function of L2 regularization strength for the max Sharpe portfolio.*

As the L2 penalty increases from zero, weight concentration (HHI) decreases monotonically toward the equal-weight level, while the ex-ante Sharpe ratio first increases (correcting for overfitting at $\gamma = 0$) before decreasing as the regularization dominates the optimization objective. The optimal $\gamma$ balances estimation error reduction against signal attenuation.

---

## Backtest Results

The table below presents annualized performance statistics from the walk-forward backtest on five years of synthetic GBM data. Values are derived from the `visualization.ipynb` notebook; see that notebook for reproduction.

| Strategy              | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD   | CVaR 95% |
|-----------------------|-------------|----------|--------|---------|----------|----------|
| Max Sharpe (BL+LW)    | 12.28%      | 7.68%    | 1.34   | 2.06    | -7.70%   | 0.91%    |
| Min Volatility        | 9.56%       | 6.94%    | 1.09   | 1.68    | -8.63%   | 0.85%    |
| Risk Parity (EWM)     | 10.33%      | 7.41%    | 1.12   | 1.72    | -9.37%   | 0.89%    |
| CVaR Optimizer        | 8.04%       | 7.31%    | 0.83   | 1.26    | -9.81%   | 0.90%    |

Values are from a walk-forward backtest on five years of synthetic GBM data (`rng seed=42`, monthly rebalancing, 252-day lookback, 10 bps proportional transaction costs). Execute `notebooks/visualization.ipynb` end-to-end to reproduce exactly.

---

## Conclusions

The walk-forward backtest on GBM-simulated data demonstrates that all four strategies achieve positive risk-adjusted returns, with the Max Sharpe (BL+LW) strategy leading at a 12.28% annualized return and a Sharpe ratio of 1.34, followed by Risk Parity (EWM) at 10.33% / 1.12, Min Volatility at 9.56% / 1.09, and CVaR Optimizer at 8.04% / 0.83. On synthetic data where the true covariance structure is known at generation time, the Ledoit-Wolf estimator and Black-Litterman returns provide a marginal edge over sample estimates because T/N = 126 is large. On real equity data with $N = 100$ assets and T = 252 daily observations, where $T/N = 2.52$, the estimation-robust approaches are expected to generate substantially larger improvements in out-of-sample Sharpe ratios relative to naive sample-based estimates.

The Black-Litterman framework's primary value in this implementation is structural rather than parametric: by anchoring expected returns to the equilibrium implied by market capitalization weights, it prevents the optimizer from acting on extreme historical return estimates that are statistically indistinguishable from noise. Combined with Ledoit-Wolf shrinkage, the joint estimation pipeline produces covariance matrices and expected return vectors whose condition numbers remain tractable across the full rolling window history, avoiding the occasional SLSQP divergence that arises when the sample covariance becomes ill-conditioned during lookback windows containing volatile subperiods.

The tension between risk concentration and diversification is clearly visible in the cross-strategy comparison. The Max Sharpe portfolio produces the shallowest maximum drawdown at -7.70%, benefiting from the tilt toward higher-Sharpe assets identified by the Black-Litterman prior. The CVaR Optimizer incurs the deepest drawdown at -9.81%, which may appear counterintuitive but reflects that minimizing the expected tail loss in the estimation window does not guarantee the minimum realized drawdown in the evaluation window. The risk parity portfolio inverts the concentration logic of mean-variance optimization, allocating capital inversely proportional to the marginal risk contribution of each asset; its diversification ratio consistently exceeds that of the Max Sharpe portfolio. The practical conclusion is that risk parity is preferable when drift estimation is unreliable (low T/N, high noise-to-signal), whereas mean-variance optimization with robust estimation dominates when reliable return forecasts are available.

The CVaR optimizer targets the expected loss in the worst $(1-\beta)$ fraction of scenarios rather than the second moment of the return distribution. For insurance-linked portfolios, pension funds operating under regulatory VaR constraints, and any mandate where tail loss is explicitly penalized, CVaR minimization is the appropriate objective. On symmetric GBM data, the CVaR and mean-variance objectives yield qualitatively similar portfolios; the difference becomes material on data exhibiting negative skewness and excess kurtosis, where the tails are heavier than Gaussian and the CVaR-optimal solution would further de-risk the tail contributors relative to the Sharpe-maximizing solution.

Future development of this library will focus on extending the estimation layer with regime-switching covariance models, the optimization layer with cardinality constraints and robust uncertainty sets, and the backtesting layer with intraday-granularity transaction cost models that incorporate asset-level average daily volume. These extensions are detailed in the Limitations section below.

---

## Limitations

### 9.1 Model Assumptions

GBM assumes constant drift $\mu_i$ and instantaneous volatility $\sigma_i$, which are empirically false for equity returns. Real asset prices exhibit volatility clustering (GARCH effects), mean-reversion in volatility (implied by option markets), and structural breaks (regime changes). Backtest results on GBM data are therefore optimistic in the sense that the return-generating process is stationary and the parameter space is stable, eliminating the distribution shift that dominates out-of-sample performance degradation in real portfolios. Asset return distributions exhibit excess kurtosis and negative skewness not captured by the Gaussian assumption underlying parametric VaR and CVaR. On real equity data, historical simulation (already implemented in `cvar_historical`) is preferable to parametric CVaR for confidence levels above 99%. The factor model implemented here uses PCA-based statistical factors, which are mathematical constructs without direct economic interpretation. Commercial models such as Barra (MSCI) use pre-specified fundamental factors — market, size, value, momentum, quality — whose loadings are estimated from cross-sectional regressions. The statistical factors in this implementation may rotate across windows, making it difficult to track factor exposure over time.

### 9.2 Estimation Error

Merton (1980) demonstrated that the variance of the portfolio return estimate is proportional to $N$ times the variance of the individual expected return estimates, while the variance attributable to covariance estimation errors is proportional to $N^2/T$. For typical portfolio sizes and observation windows, expected return estimation error dominates. The Black-Litterman framework mitigates this by shrinking $\hat{\boldsymbol{\mu}}$ toward $\boldsymbol{\Pi}$, but it does not eliminate the sensitivity: the posterior is still a linear function of the sample covariance, which is itself estimated with error. The shrinkage intensity $\alpha$ in Ledoit-Wolf OAS is estimated from the same data used to construct the covariance matrix, introducing a subtle overfitting bias in short rolling windows. A holdout-based calibration of the shrinkage target would reduce this bias at the cost of additional data requirements. The Black-Litterman Omega (view uncertainty matrix) is approximated as $\text{diag}(\tau\mathbf{P}\boldsymbol{\Sigma}\mathbf{P}^\top)$ following Idzorek's method, which assumes views are uncorrelated. A full Bayesian treatment with elicited off-diagonal view covariances would require additional practitioner input that is not feasible in a fully automated pipeline.

### 9.3 Optimization Limitations

SLSQP is a local gradient-based solver. For the maximum Sharpe ratio objective, which is non-convex in $\mathbf{w}$ (despite the Charnes-Cooper transformation), the multi-restart strategy with five Dirichlet initializations reduces but does not eliminate the probability of terminating at a non-global local optimum, particularly in high-dimensional problems ($N \ge 50$) or under tight constraint sets. The turnover constraint is implemented via proportional weight shrinkage toward the current portfolio: $\mathbf{w}_{\text{constrained}} = \mathbf{w}_{\text{old}} + \lambda(\mathbf{w}_{\text{new}} - \mathbf{w}_{\text{old}})$, where $\lambda = \min(1, L_{\max}/L)$ and $L$ is the realized one-way turnover. This is a first-order approximation to the true L1-constrained optimization problem and does not guarantee optimality of the constrained solution. Cardinality constraints — for example, "hold at most 20 of 100 available assets" — require mixed-integer quadratic programming (MIQP) and are not supported; the library assumes continuous weight allocations throughout.

### 9.4 Backtesting Limitations

The synthetic GBM data does not exhibit autocorrelation in returns, GARCH-type volatility clustering, or structural breaks. Walk-forward backtests on GBM data therefore overstate the reliability of out-of-sample performance, because the lookback-window parameters remain in-distribution for the evaluation window. On real equity and credit data, out-of-sample Sharpe ratios are typically 30–60% lower than in-sample estimates due to parameter instability. No survivorship bias correction is possible on synthetic data. Applied to real data, point-in-time constituent lists (as opposed to current index constituents) are required to avoid the well-documented survivorship bias in historical backtests. The transaction cost model assumes proportional spreads independent of trade size. In practice, market impact is a superlinear function of trade size relative to average daily volume (ADV). Modeling this properly requires asset-level ADV data and an impact model (e.g., Almgren-Chriss, 2001) that is not incorporated in the current implementation. No slippage or execution delay is modeled. A realistic implementation shortfall model would apply a one-day lag between the signal date and the execution date.

### 9.5 Missing Features (Planned Extensions)

The following methods are not currently implemented but are identified as high-priority extensions:

- **Hierarchical Risk Parity (HRP):** Ledoit-Wolf shrinkage combined with single-linkage clustering and recursive bisection (López de Prado, 2016). HRP does not require matrix inversion and is more robust to degenerate covariance estimates.
- **Regime-switching covariance:** Hidden Markov Model on realized volatility regime to switch between a low-volatility and high-volatility covariance matrix, with Viterbi decoding for regime classification.
- **Online covariance updates:** Sherman-Morrison-Woodbury rank-one covariance updates in $O(N^2)$ per period, enabling streaming backtests without full re-estimation at each step.
- **Cardinality-constrained optimization:** Mixed-integer quadratic programming (MIQP) to enforce a maximum number of held positions, using branch-and-bound solvers such as GUROBI or CVXPY's SCIP interface.
- **Robust optimization under ellipsoidal uncertainty:** Ben-Tal and Nemirovski (1998) uncertainty sets centered on the point estimate $\hat{\boldsymbol{\mu}}$, yielding tractable second-order cone programs (SOCP) that are immune to worst-case parameter realizations.
- **Multi-period dynamic portfolio optimization:** Transaction-cost-regularized stochastic dynamic programming over a finite horizon, generalizing the single-period objective to account for the intertemporal tradeoff between rebalancing costs and tracking error from the optimal static portfolio.

---

## References

Markowitz, H. M. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.

Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28–43.

He, G., & Litterman, R. (1999). *The intuition behind Black-Litterman model portfolios*. Goldman Sachs Investment Management Division.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.

Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance estimation. *IEEE Transactions on Signal Processing*, 58(10), 5016–5029.

Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41.

Roncalli, T. (2013). *Introduction to risk parity and budgeting*. CRC Press.

Merton, R. C. (1980). On estimating the expected return on the market: An exploratory investigation. *Journal of Financial Economics*, 8(4), 323–361.

Ben-Tal, A., & Nemirovski, A. (1998). Robust convex optimization. *Mathematics of Operations Research*, 23(4), 769–805.

Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39.

MSCI Barra. (2011). *Barra risk model handbook*. MSCI.

López de Prado, M. (2016). Building diversified portfolios that outperform out-of-sample. *Journal of Portfolio Management*, 42(4), 59–69.
