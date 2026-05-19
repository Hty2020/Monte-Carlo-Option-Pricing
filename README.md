# Monte Carlo Option Pricer

A Python implementation of Monte Carlo pricing for European call options under two models: the classic **Black-Scholes / GBM** model and the **Heston stochastic-volatility** model. The project covers the full numerical workflow — simulation, variance reduction, performance optimisation, convergence analysis, and empirical benchmarking against real market data.

---

## Models

### Black-Scholes (GBM)

The stock price $S_t$ follows Geometric Brownian Motion:

$$dS_t = r S_t  dt + \sigma S_t  dW_t$$

which has the exact solution

$$S_T = S_0 \exp\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma W_T\right]$$

Two simulation modes are implemented:

- **Path simulation** (`mc_simulation`): discretises the SDE step-by-step in log-space to avoid Itô-discretisation bias; produces full price trajectories.
- **Direct sampling** (`direct_sampling`): exploits the exact GBM solution to draw $S_T$ directly, bypassing time-stepping entirely.

Both modes support **antithetic variates**, which pairs each Gaussian draw $z$ with $-z$, halving the variance of the estimator at no extra simulation cost (~2× standard-error reduction observed empirically).

Prices are validated against the closed-form Black-Scholes formula implemented in `analytical_solution`.

### Heston Stochastic-Volatility Model

The Heston model relaxes the constant-volatility assumption by letting the variance $v_t$ follow a mean-reverting CIR process correlated with the stock:

$$dS_t = r S_t  dt + \sqrt{v_t}  S_t  dW_t^S$$

$$dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t}  dW_t^v, \qquad d W^S  dW^v = \rho dt$$

| Parameter | Meaning |
|-----------|---------|
| $v_0$ | Initial variance |
| $\kappa$ | Mean-reversion speed |
| $\theta$ | Long-run variance |
| $\xi$ | Variation of variations |
| $\rho$ | Correlation between price and variance |

The two correlated Wiener increments are constructed as follows:

$$W^v = \rho W^S + \sqrt{1-\rho^2} W^v, \qquad W^v \perp W^S$$

Simulation uses log-space discretisation with absolute-value reflection to keep variance non-negative.

Antithetic variates are applied to both Wiener processes simultaneously. Prices are validated against the semi-analytical Heston price computed via characteristic-function integration:

$$C = S_0 P_1 - K e^{-rT} P_2, \qquad P_j = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \mathrm{Re}\left[\frac{e^{-i\phi \ln K} f_j(\phi)}{i\phi}\right] d\phi$$

---

## Performance

Both classes accept a `USE_NUMBA` flag. When enabled, the inner simulation loops are compiled to native machine code via **Numba JIT** (`@jit(nopython=True)`). The Heston simulator additionally uses `parallel=True` for multi-core execution.

> **Note:** The first call with Numba incurs a one-time JIT compilation overhead.

---

## Empirical Results

### Black-Scholes limitations

The constant-$\sigma$ assumption was tested empirically on real return data by segmenting one year of daily log-returns into 20 sub-intervals and computing the local standard deviation in each. Key findings:

- $\sigma$ varies substantially across sub-intervals — the constant-volatility assumption does not hold.
- The distribution of log-returns exhibits **fat tails**, deviating from the Gaussian assumption.

### Model comparison vs. market data

Both models were benchmarked against live option chains (via `yfinance`) for three tickers spanning different volatility regimes:

| Ticker | Description |
|--------|-------------|
| AAPL | Large company|
| GOOGL | Large company|
| CRSP | Mid biotech |

For each ticker, prices were compared across 5 expiry dates (fixed strike) and across 5 strikes (fixed expiry).

**BS model**: MC agrees with the analytical solution (no systematic bias), but both shows noticeable deviations from real market prices.

**Heston model**: MC agrees with the semi-analytical price. However, with hand-picked parameters the model does not fit market prices well — proper calibration is required. One calibration objective is:

$$\min_{\kappa,\,\theta,\,\xi,\,\rho,\,v_0} \sum_i \left(\sigma_i^{\text{Heston}} - \sigma_i^{\text{Market}}\right)^2$$

After calibration, one sees a better much in volatility curve as well as a better accuracy.

---

## Repository Structure

```
.
├── MC_Pricer.py              # Core classes: MonteCarlo_GBM, MonteCarlo_Heston
|── Heston_Calibration.py             # the tool box used for calibrating Heston model
└── MC_Option_Pricer_2.ipynb    # Full analysis notebook
```

### `MC_Pricer.py` — Public API

**`MonteCarlo_GBM(USE_NUMBA=True)`**

| Method | Description |
|--------|-------------|
| `load(S0, K, sigma, r_free)` | Set model parameters |
| `mc_simulation(T, n_samples, n_steps)` | Path simulation; returns `(price, trajectories)` |
| `direct_sampling(T, n_samples, antithetic)` | Terminal-value sampling; returns `(price, std_error)` |
| `analytical_solution(T)` | Closed-form Black-Scholes price |

**`MonteCarlo_Heston(USE_NUMBA=True)`**

| Method | Description |
|--------|-------------|
| `load(S0, K, r_free, v0, kappa, theta, xi, rho)` | Set model parameters |
| `mc_simulation(T, n_samples, n_steps, antithetic)` | Path simulation; returns `(price, std_error)` |
| `mc_demo(T, n_samples, n_steps)` | Returns full `(S_trajectories, v_trajectories)` for visualisation |
| `heston_price(T)` | Semi-analytical price via characteristic-function integration |

---

## Usage Example

```python
from MC_Pricer import MonteCarlo_GBM, MonteCarlo_Heston

# --- Black-Scholes ---
gbm = MonteCarlo_GBM(USE_NUMBA=True)
gbm.load(S0=175.0, K=180.0, sigma=0.25, r_free=0.037)

price, std_err = gbm.direct_sampling(T=1.0, n_samples=1_000_000, antithetic=True)
bs_price       = gbm.analytical_solution(T=1.0)

print(f"MC price : {price:.4f} ± {2*std_err:.4f} (2σ)")
print(f"BS price : {bs_price:.4f}")

# --- Heston ---
heston = MonteCarlo_Heston(USE_NUMBA=True)
heston.load(S0=175.0, K=180.0, r_free=0.037,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5)

price, std_err   = heston.mc_simulation(T=1.0, n_samples=500_000, n_steps=500, antithetic=True)
analytical_price = heston.heston_price(T=1.0)

print(f"Heston MC price         : {price:.4f} ± {2*std_err:.4f} (2σ)")
print(f"Heston analytical price : {analytical_price:.4f}")
```

---

## Limitations and Next Steps

- **Heston calibration**: Parameters are currently set manually. Fitting data via least-squares optimisation is the natural next step.
- **Discretisation scheme**: The direct time discretization scheme with absolute-value reflection is simple but may not be the most accurate for the CIR variance process. Probably one could use the method inspired from molecular dynamics simulation (to be checked).
- **Scope**: Only European call options are priced. Path-dependent payoffs would require the full trajectory and would be a natural extension.
