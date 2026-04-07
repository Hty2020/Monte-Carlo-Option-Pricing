# Monte Carlo Pricer
import numpy as np
from numba import jit

class MonteCarlo:
    # Base class 
    # Monte Carlo simulator for one specific call option

    def __init__(self):
        self._S0 = None         # initial stock price
        self._K = None          # strike price
        self._r_free = None     # risk-free rate

    def load(self, *args):
        raise NotImplementedError("Should be implemented in the child class.")
    
    def mc_simulation(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the child class.")
    
class MonteCarlo_GBM(MonteCarlo):
    def __init__(self, USE_NUMBA=True):
        super().__init__()

        if USE_NUMBA:
            self._direct_sampling = jit(nopython=True)(self._direct_sampling_impl)
            self._mc_simulation = jit(nopython=True)(self._mc_simulation_impl)
        else:
            self._direct_sampling = self._direct_sampling_impl
            self._mc_simulation = self._mc_simulation_impl
        

    def load(self, S0, K, sigma, r_free):
        self._S0 = S0
        self._K = K
        self._sigma = sigma
        self._r_free = r_free

    @staticmethod
    def _mc_simulation_impl(T, S0, r, sigma, K, n_samples=5, n_steps=100):
        trajectory = np.empty((n_samples, n_steps+1))
        dt = T / n_steps
        
        for i in range(n_samples):       # (!)not fully optimized: different realizations can be further vectorized
            log_S = np.log(S0)           # work in log-space to avoid discretization error from It\^o's lemma
            trajectory[i, 0] = S0
            for step in range(1, n_steps+1):
                log_S += (r - 0.5*(sigma**2)) * dt + sigma*np.sqrt(dt)*np.random.randn()
                trajectory[i, step] = np.exp(log_S)

        S_final = trajectory[:, -1].copy()
        payoffs = np.maximum(S_final - K, 0)

        return np.exp(-r * T) * np.mean(payoffs), trajectory
    
    def mc_simulation(self, T, n_samples=5, n_steps=100):
        return self._mc_simulation(T, self._S0, self._r_free, self._sigma, self._K, n_samples, n_steps)

    @staticmethod
    def _direct_sampling_impl(T, n_samples, S0, r, sigma, K, antithetic=True):
        # based on the analytical solution of GBM, directly making MC averaging
        gaussians = np.random.randn(n_samples)
        if antithetic:
            S_p = S0 * np.exp((r - 0.5*sigma**2)*T + sigma * np.sqrt(T) * gaussians)
            S_m = S0 * np.exp((r - 0.5*sigma**2)*T - sigma * np.sqrt(T) * gaussians)
            payoffs = (np.maximum(S_p - K, 0) + np.maximum(S_m - K, 0)) / 2
        else:
            S = S0 * np.exp((r - 0.5*sigma**2)*T + sigma * np.sqrt(T) * gaussians)
            payoffs = np.maximum(S - K, 0)

        return np.exp(-r * T) * np.mean(payoffs), np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_samples)

    def direct_sampling(self, T, n_samples, antithetic=True):
        return self._direct_sampling(T, n_samples, self._S0, self._r_free, self._sigma, self._K, antithetic)
    
    def analytical_solution(self, T):
        from scipy.stats import norm
        d1 = (np.log(self._S0/self._K) + (self._r_free + 0.5 * self._sigma**2) * T) / (self._sigma * np.sqrt(T))
        d2 = d1 - self._sigma * np.sqrt(T)
        C_BS = self._S0 * norm.cdf(d1) - self._K * np.exp(-self._r_free * T) * norm.cdf(d2)
        return C_BS
    

class MonteCarlo_Heston(MonteCarlo):
    def __init__(self, USE_NUMBA=True):
        super().__init__()

        self._v0 = None
        self._kappa = None
        self._theta = None
        self._xi = None
        self._rho = None

        if USE_NUMBA:
            self._mc_simulation = jit(nopython=True, parallel=True)(self._mc_simulation_impl)
        else:
            self._mc_simulation = self._mc_simulation_impl
        
    def load(self, S0, K, r_free, v0, kappa, theta, xi, rho):
        self._S0 = S0
        self._K = K
        self._r_free = r_free
        self._v0 = v0
        self._kappa = kappa
        self._theta = theta
        self._xi = xi
        self._rho = rho

    def mc_demo(self, T, n_samples=5, n_steps=100):
        # show demostrations
        # storing the path of each realization, hence memory consuming; only for demo
        dt = T / n_steps
        
        S_trajectories = np.empty((n_samples, n_steps+1))
        S_trajectories[:, 0] = self._S0
        v_trajectories = np.empty((n_samples, n_steps+1))
        v_trajectories[:, 0] = self._v0

        log_S = np.ones(n_samples) * np.log(self._S0)
        v = np.ones(n_samples) * self._v0

        for step in range(1, n_steps+1):
            gaussians_1 = np.random.randn(n_samples)
            gaussians_2 = np.random.randn(n_samples)

            W_S = gaussians_1
            W_v = self._rho * gaussians_1 + np.sqrt(1 - self._rho**2) * gaussians_2 # correlate the two Wiener processes

            # A simple yet direct calculator; might be possible to improve the accuracy by using the techniques in molecular dynamics PDE simulation
            v = np.abs(v)       # Uhlenbeck process
            log_S = log_S + (self._r_free - 0.5*v) * dt + np.sqrt(v*dt) * W_S
            v = v + (self._kappa * (self._theta-v)) * dt + self._xi * np.sqrt(v*dt) * W_v

            
            S_trajectories[:, step] = np.exp(log_S)
            v_trajectories[:, step] = v

        return S_trajectories, v_trajectories

    @staticmethod
    def _mc_simulation_impl(T, S0, r, K, v0, kappa, theta, xi, rho, n_samples, n_steps, antithetic):
        dt = T / n_steps

        # calculated the needed parameters once for all
        sqrt_dt = np.sqrt(dt)
        sqrt_one_minus_rho2 = np.sqrt(1.0 - rho * rho)
        log_S0 = np.log(S0)

        payoffs = np.empty(n_samples)

        if antithetic:
            for i in range(n_samples):
                # initialization
                log_S = log_S0
                v = v0
                log_S_m = log_S0
                v_m = v0

                for _ in range(n_steps):
                    gaussian1 = np.random.randn()
                    gaussian2 = np.random.randn()
                    W_S = gaussian1
                    W_v = rho * gaussian1 + sqrt_one_minus_rho2 * gaussian2

                    v = np.abs(v)       # Ornstein-Uhlenbeck process
                    sqrt_v = np.sqrt(v)
                    v_m = np.abs(v_m)
                    sqrt_v_m = np.sqrt(v_m)

                    log_S = log_S + (r - 0.5 * v) * dt + sqrt_v * sqrt_dt * W_S
                    v = v + kappa * (theta - v) * dt + xi * sqrt_v * sqrt_dt * W_v
                    log_S_m = log_S_m + (r - 0.5 * v_m) * dt - sqrt_v_m * sqrt_dt * W_S
                    v_m = v_m + kappa * (theta - v_m) * dt - xi * sqrt_v_m * sqrt_dt * W_v

                payoff  = np.maximum(np.exp(log_S) - K, 0)
                payoff_antithetic = np.maximum(np.exp(log_S_m) - K, 0)
        
                payoffs[i] = 0.5 * (payoff + payoff_antithetic)
        else:
            for i in range(n_samples):
                # initialization
                log_S = np.log(S0) 
                v = v0

                for _ in range(n_steps):
                    gaussian1 = np.random.randn()
                    gaussian2 = np.random.randn()
                    W_S = gaussian1
                    W_v = rho * gaussian1 + np.sqrt(1.0 - rho * rho) * gaussian2

                    v = np.abs(v)       # Ornstein-Uhlenbeck process
                    log_S = log_S + (r - 0.5 * v)  * dt + np.sqrt(v)  * np.sqrt(dt)* W_S
                    v = v + kappa * (theta - v)  * dt + xi * np.sqrt(v * dt) * W_v

                payoffs[i] = np.maximum(np.exp(log_S) - K, 0)

        discount = np.exp(-r * T)
        return discount * np.mean(payoffs), discount * np.std(payoffs) / np.sqrt(n_samples)

    
    def mc_simulation(self, T, n_samples=5, n_steps=100, antithetic=True):
        return self._mc_simulation(T, self._S0, self._r_free, self._K, self._v0, self._kappa, self._theta, self._xi, self._rho, n_samples, n_steps, antithetic)

    @staticmethod
    def heston_char_func(phi, S0, v0, kappa, theta, xi, rho, r, T, j):
        """Characteristic function for Heston model (j=1 or 2)."""
        if j == 1:
            u = 0.5;  b = kappa - rho*xi
        else:
            u = -0.5; b = kappa

        a = kappa * theta
        x = np.log(S0)
        d = np.sqrt((rho*xi*phi*1j - b)**2 - xi**2*(2*u*phi*1j - phi**2))
        g = (b - rho*xi*phi*1j + d) / (b - rho*xi*phi*1j - d)

        C = r*phi*1j*T + (a/xi**2)*((b - rho*xi*phi*1j + d)*T
            - 2*np.log((1 - g*np.exp(d*T))/(1 - g)))
        D = ((b - rho*xi*phi*1j + d)/xi**2) * ((1 - np.exp(d*T))/(1 - g*np.exp(d*T)))

        return np.exp(C + D*v0 + 1j*phi*x)

    def heston_price(self, T):
        from scipy.integrate import quad

        def integrand(phi, j):
            cf = MonteCarlo_Heston.heston_char_func(phi, self._S0, self._v0, self._kappa, self._theta, self._xi, self._rho, self._r_free, T, j)
            return np.real(np.exp(-1j*phi*np.log(self._K)) * cf / (1j*phi))

        P1 = 0.5 + (1/np.pi) * quad(lambda phi: integrand(phi, 1), 0, 100)[0]
        P2 = 0.5 + (1/np.pi) * quad(lambda phi: integrand(phi, 2), 0, 100)[0]

        return self._S0 * P1 - self._K * np.exp(-self._r_free * T) * P2