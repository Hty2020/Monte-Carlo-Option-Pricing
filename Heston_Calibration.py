# Calibrate the Heston Model
import numpy as np
from scipy.stats import norm

from MC_Pricer import black_scholes_call, heston_call

def bs_implied_vol(price, S0, K, T, r):
    """Invert BS formula numerically to get implied vol."""
    from scipy.optimize import brentq
    def objective(sigma):
        return black_scholes_call(S0, K, T, r, sigma) - price
    try:
        return brentq(objective, 1e-6, 10.0)
    except:
        return np.nan

def calibration_loss(params, market_data, S0, r):
    v0, kappa, theta, xi, rho = params

    # Enforce constraints — return large penalty instead of raising so Nelder-Mead can recover
    if v0 <= 0 or kappa <= 0 or theta <= 0 or xi <= 0:
        return 1e6
    if not (-1 < rho < 0):   # enforce negative correlation
        return 1e6

    total_loss = 0
    for K, T, sigma_mkt in market_data:
        price = heston_call(S0, K, T, r, v0, kappa, theta, xi, rho)
        sigma_model = bs_implied_vol(price, S0, K, T, r)
        if not np.isnan(sigma_model):
            total_loss += (sigma_model - sigma_mkt)**2
    return total_loss


if __name__ == '__main__':
    # test
    import yfinance as yf
    import pandas as pd
    from scipy.optimize import minimize

    r_free=3.673e-2 # the three-month US Treasury yield (19-05-2026)

    ticker = yf.Ticker("AAPL")
    expiries = ticker.options   # list of available expiry dates
    S0 = ticker.fast_info["last_price"]
    today = pd.Timestamp.today().normalize()

    # Flatten into (K, T, sigma_mkt) tuples with T = time-to-maturity in years
    market_data = []
    for expiry in expiries[:5]:   # use first 5 expiries
        chain = ticker.option_chain(expiry)
        calls = chain.calls[["strike", "impliedVolatility"]].dropna()

        expiry_date = pd.Timestamp(expiry)
        n_trading_days = len(pd.bdate_range(today, expiry_date))
        T = n_trading_days / 252
        if T <= 0:
            raise ValueError('The expiry date should be later than today!')
        
        for _, row in calls.iterrows():
            K = row["strike"]
            sigma_mkt = row["impliedVolatility"]
            if sigma_mkt > 0 and K <= S0:
                market_data.append((K, T, sigma_mkt))


    result = minimize(calibration_loss, x0=[0.04, 1.5, 0.06, 0.5, -0.5], args=(market_data, S0, r_free), method='Nelder-Mead')
    v0, kappa, theta, xi, rho = result.x

    print(f"v0 = {v0:.4f}  (sigma0 = {np.sqrt(v0)*100:.1f}%)")
    print(f"kappa = {kappa:.4f} (vol half-life = {np.log(2)/kappa*252:.0f} trading days)")
    print(f"theta = {theta:.4f} (long-run sigma = {np.sqrt(theta)*100:.1f}%)")
    print(f"xi = {xi:.4f}  (vol-of-vol)")
    print(f"rho = {rho:.4f}  (leverage effect)")
    print(f"Feller condition 2κθ/ξ² = {2*kappa*theta/xi**2:.3f}  (>1 for non-zero variance)")
    print(f"Converged: {result.success}, Loss: {result.fun:.6f}")