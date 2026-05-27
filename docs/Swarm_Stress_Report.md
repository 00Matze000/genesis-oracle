# Swarm Stress Report: JAX Monte Carlo Performance & Sensitivity Analysis

This report synthesizes the analytical outputs of the automated subagents **Subagent-Alpha ('The Stress-Tester')** and **Subagent-Beta ('The Profiler')** evaluating the stochastically parallelized `src/monte_carlo.py` pipeline.

---

## 1. Executive Summary

By leveraging JAX's functional purity and compilation pipeline, we simulated $1,000,000$ business trajectories in parallel. The baseline configuration demonstrates an extremely robust economic outlook, with an expected revenue of **$149,751.72$ EUR** and a $95\%$ confidence limit (Value-at-Risk) of **$112,734.47$ EUR**. 

Through swarm-based sensitivity sweeps and performance profiling, we identified:
- The **breaking point** of the economic model under cost volatility: **$\sigma_C \approx 4.00$**.
- The **performance acceleration** achieved via XLA compilation: **$7.00\times$ speedup**.

---

## 2. Sensitivity Analysis (Subagent-Alpha)

**Subagent-Alpha ('The Stress-Tester')** programmatically swept the variance parameter ($\sigma_C$) of the Log-Normal production asset cost distribution:
$$\ln(C) \sim \mathcal{N}(\mu = 5.5, \sigma^2 = \sigma_C^2)$$

The goal was to identify the exact threshold where the $VaR_{95\%}$ (the 5th percentile of net revenue) drops below zero, representing a structural threat of loss under high-risk scenarios.

### Parameter Sweep Results

| Variance Parameter ($\sigma_C$) | Expected Revenue $E[\text{Revenue}]$ (EUR) | Value-at-Risk $VaR_{95\%}$ (EUR) | Status / Risk Assessment |
| :---: | :---: | :---: | :---: |
| $0.30$ (Baseline) | $+149,751.72$ | $+112,734.47$ | **Highly Profitable / Minimal Risk** |
| $3.00$ | $+132,772.56$ | $+98,868.55$ | Robust operational buffer |
| $3.80$ | $-65,402.70$ | $+39,655.66$ | E[Revenue] negative; VaR still buffered |
| $3.90$ | $-154,256.72$ | $+21,464.62$ | Highly volatile, severe tail-end stress |
| $3.99$ | $-267,173.78$ | $+2,047.15$ | Critical threshold border |
| **$4.00$ (Breaking Point)** | **$-282,180.97$** | **$-333.83$** | **Loss Event Triggered ($VaR_{95\%} < 0$)** |

### Statistical Insights

1. **The Log-Normal Explosion:** The expected value of a Log-Normal variable is heavily determined by its variance:
   $$\mathbb{E}[C] = e^{\mu + \frac{\sigma_C^2}{2}}$$
   As $\sigma_C$ increases, the right tail of the cost distribution stretches exponentially. At $\sigma_C = 4.00$, the expected costs balloon to approximately $729,000$ EUR, pushing the expected net revenue deep into negative territory ($-282,180.97$ EUR).
2. **The VaR Resilience:** Despite the negative expected revenue at $\sigma_C = 3.80$, the $VaR_{95\%}$ remains positive. This behavior is a direct consequence of the log-normal distribution's extreme right-skewness: $95\%$ of all cost paths remain small, keeping the corresponding revenues positive, while a tiny fraction ($5\%$) of catastrophic cost spikes drags down the expectation. Only at **$\sigma_C \approx 4.00$** does the volatility finally overwhelm the $95\%$ confidence barrier.

---

## 3. Performance Profiling (Subagent-Beta)

**Subagent-Beta ('The Profiler')** evaluated the computational overhead of JAX's compilation phase versus the vectorized warm-execution pass. The simulation evaluated $1,000,000$ complete paths in parallel.

### Execution Metrics

*   **First Run (Cold Execution / Compilation + Tracing):** **$1.039856$ seconds**
*   **Second Run (Warm Execution / Compiled XLA Code):** **$0.148586$ seconds**
*   **Vectorized Speedup Factor:** **$7.00\times$**

### Technical Analysis

*   **The Tracing Phase:** During the first execution, JAX wraps the Python function `simulate_path` in an abstract representation by executing it with placeholder variables (tracing). This representation is compiled into highly optimized machine instructions via XLA (Accelerated Linear Algebra).
*   **The Warm Pass:** In the second execution, the compiled code runs directly on the CPU/GPU registers. The Python overhead is entirely bypassed, executing 1 million paths in a fraction of a second ($0.148$s). This showcases the immense power of `jax.vmap` combined with `@jit` pipelines.
