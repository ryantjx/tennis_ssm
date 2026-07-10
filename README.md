# State-Space Models for WTA matches

## Model

Following Duffield, Power and Rimella (2024), the model for tennis can be defined as follows:

$$p(x_0) \sim \mathcal{N}(\mu_0, \Sigma_0)$$

$$p(x_t | x_{t-1}) \sim \mathcal{N}(\mu_0 + \phi_k (x_{t-1} - \mu_0), Q_k)$$

where $\phi_k = \exp(-\tau_d \cdot \Delta t)$ and $Q_k = \Sigma_0 - \phi_k \Sigma_0 \phi_k^T$.

$$G_t (y_t \mid x^{(i)}_{t-1}, x^{(j)}_{t-1}) = \begin{cases} \sigma(\frac{x^{(i)} - x^{(j)}}{s_d}) & \text{if } y_t = x^{(i)} \\ 1 - \sigma(\frac{x^{(i)} - x^{(j)}}{s_d}) & \text{if } y_t = x^{(j)} \end{cases}$$

where:
- $y_t$ is the outcome of match $t$ between players $i$ and $j$
- $\sigma(x) = (1 + e^{-x})^{-1}$ is the sigmoid function; larger $s_d$ flattens the win probability toward uniform (0.5)
- $\tau_d \in \mathbb{R}^+$ is the mean-reversion rate
- $s_d \in \mathbb{R}^+$ is the observation scale

We fix $\mu_0 = 0$ so skills are centered at zero. The parameters to estimate are $\Sigma_0, \tau_d, s_d \in \mathbb{R}^+$.

Unlike TrueSkill2 and Elo, which use Wiener processes (unbounded variance growth), this model uses an Ornstein-Uhlenbeck process where skills revert to $\mathcal{N}(\mu_0, \Sigma_0)$ during inactive periods.

## Algorithm

We use the Gaussian Moments filter from `cuthbert` to perform inference. We assume that the distribution of player skills is independent from another player given the match outcome, so we can factorize the joint posterior of two players $i$ and $j$ as follows:

$$p(x^{(i)}_t, x^{(j)}_t | y_{1:t}) \propto p(x^{(i)}_t \mid y_{1:t}) p(x^{(j)}_t \mid y_{1:t})$$

Our next assumption that enables the Gaussian Moments filter is that the posterior of each player skill can be approximated as a Gaussian distribution,

$$p(x^{(i)}_t \mid y_{1:t}) \approx \mathcal{N}(\mu^{(i)}_t, \Sigma^{(i)}_t)$$

The Gaussian Moments filter proceeds in two steps:

**1. Prediction:** Between matches, player skills evolve according to the state transition model:

  $$\mu_{t|t-1}^{(i)} = \mu_0 + \phi_k (\mu_{t-1}^{(i)} - \mu_0)$$

  $$\Sigma_{t|t-1}^{(i)} = \phi_k \Sigma_{t-1}^{(i)} \phi_k^T + Q_k$$

**2. Update:** After observing a match outcome, the exact posterior is non-Gaussian due to the sigmoid likelihood. The Gaussian Moments filter approximates this posterior by matching its first two moments (mean and covariance), yielding the standard Kalman filter update equations:

$$\mu_t^{(i)} = \mu_{t|t-1}^{(i)} + K_t^{(i)} (y_t - G_t(\mu_{t|t-1}^{(i)}, \mu_{t|t-1}^{(j)}))$$

$$K_t^{(i)} = \Sigma_{t|t-1}^{(i)} H_t^T (H_t \Sigma_{t|t-1}^{(i)} H_t^T + R_t)^{-1}$$

$$\Sigma_t^{(i)} = (I - K_t^{(i)} H_t) \Sigma_{t|t-1}^{(i)}$$

where $H_t$ is the Jacobian of $G_t$ with respect to $x^{(i)}$ evaluated at the predicted means $\mu_{t|t-1}^{(i)}$ and $\mu_{t|t-1}^{(j)}$, and $R_t$ is the observation noise covariance (for binary outcomes, this is derived from the Bernoulli variance $G_t(1-G_t)$).

## Prediction

To perform prediction, we use the parameters from the latest filtered state of the player's skills, $(x^{(i)}_t$, $x^{(j)}_t)$, and apply the likelihood function $G$ to compute the probability of each player winning against their future opponents.

$$G_{t+1} (y_{t+1} \mid x^{(i)}_t, x^{(j)}_t) = \begin{cases} \sigma(\frac{x^{(i)}_t - x^{(j)}_t}{s_d}) & \text{if } y_{t+1} = x^{(i)}_t \\ 1 - \sigma(\frac{x^{(i)}_t - x^{(j)}_t}{s_d}) & \text{if } y_{t+1} = x^{(j)}_t \end{cases}$$

## Data

Data is collected from http://www.tennis-data.co.uk/alldata.php.

## Results

The deployment model uses saved parameters from
`outputs/tennis_factorial_state.json`, then reruns the filter over every completed WTA historical result available in the latest data pull. This keeps training and state updates restricted to observed match outcomes. Current or unplayed fixtures are never added to the results artifact and are not used as observations.

### Performance Comparison

We performed a comparison of the Wiener process (Random Walk) and the OU process for the transition distribution that models the evolution of player skills.

**Wiener Process:**
$$p(x_t \mid x_{t-1}) \sim \mathcal{N}(x_{t-1}, \tau^2 \Delta t)$$
where variance grows linearly with time and skills undergo a random walk without mean reversion.

**Ornstein-Uhlenbeck Process:**
$$p(x_t \mid x_{t-1}) \sim \mathcal{N}(\mu_0 + \phi_k (x_{t-1} - \mu_0), Q_k)$$
where $\phi_k = \exp(-\tau_d \cdot \Delta t)$ and $Q_k = \Sigma_0 - \phi_k \Sigma_0 \phi_k^T$.

| Model | Date | $\tau$ | $s$ | $\Sigma_0$ | Accuracy | Avg Log-Score |
|-------|------|---------|-----|------------|----------|---------------|
| Wiener Process | 2026-07-10 | 0.029 | 1.76 | 1.71 | 63.1% | -0.6378 |
| OU Process (untrained) | 2026-07-10 | 0.0238 | 1.96 | 1.65 | 57.3% | -0.6778 |
| **OU Process (trained)** | **2026-07-10** | **0.000267** | **1.228** | **1.242** | **63.84%** | **-0.631** |

**Key Findings:**
- Training is essential for OU dynamics — untrained OU parameters performed poorly (57.3% vs 63.1%)
- After training, OU achieves **63.84% accuracy**, slightly exceeding the Wiener baseline
- Better log-score: **-0.631** vs -0.6378 (closer to 0 is better)
- The small τ (0.000267) indicates slow mean-reversion — skills change gradually over time

### Current Model Metrics

| Metric | Value |
|-------|-------|
| Training matches | 7,486 |
| Test matches | 1,422 |
| Accuracy | 63.84% |
| Avg log-score | -0.631 |
| Uniform baseline | -0.6931 |
**Trained parameters (OU):** $\tau=0.000267$, $s=1.228$, $\Sigma_0=1.242$

### Predictions Output

`predictions.json` contains model forecasts. It includes a completed 2026 forecast archive plus `future_matches` for unplayed fixtures loaded from the WTA fixture API. Future fixtures are predictions only; they do not update player skills until their completed results later appear in the historical results source.

Before future predictions are generated, the latest filtered state is synchronized to the newest completed match timestamp. Player rankings and future fixture predictions therefore reflect all completed historical observations available at generation time.

Future predictions are loaded from the WTA fixture API via `src/data/fixtures_womens.py` and filtered to players known by the trained model. Synthetic top-player matchups are disabled in the main workflow.

Daily generation writes the newest deployable data to:

- `outputs/latest/predictions.json`
- `outputs/latest/results.json`

It also stores dated snapshots under:

- `outputs/daily/YYYY-MM-DD/predictions.json`
- `outputs/daily/YYYY-MM-DD/results.json`

## Limitations

Currently the model only works with WTA matches. ATP (men's) matches are not supported because obtaining future data for ATP matches is not as straightforward.

On top of that historical data for WTA matches is limited to tournaments that have been completed. For example, for predicting Wimbledon games, the training data is limited up to Eastbourne Open, which was the last completed tournament before Wimbledon. Games during the Wimbledon tournament were not included in the training data. **This will affect the accuracy of predictions in later stages such as semi-finals and finals.**

## Extensions

Current model only models outcome of `{win, loss}` and does not account for any information about the sets won in the likelihood $G$.

A solution to this would be to use **independent Bernoulli for likelihood for a set won** and then use a exhaustive summation for the combination of sets. (For bo3, it would be 2-1, 2-0, 1-2, 0-2). Or an **ordered probit model** where we just model likelihood of score difference (e.g. 2-0 is +2. then combination would be +2, +1, -1, -2).

## References

- Samuel Duffield, Samuel Power, Lorenzo Rimella, A state-space perspective on modelling and inference for online skill rating, Journal of the Royal Statistical Society Series C: Applied Statistics, Volume 73, Issue 5, November 2024, Pages 1262–1282, https://doi.org/10.1093/jrsssc/qlae035
  - [SamDuffield/abile](https://github.com/SamDuffield/abile)
  <!-- - data: https://github.com/SamDuffield/abile/blob/main/datasets/tennis.py
  - train: https://github.com/SamDuffield/abile/blob/main/simulations/tennis_train.py
  - model: https://github.com/SamDuffield/abile/blob/main/abile/models/extended_kalman.py -->
- [state-space-models/cuthberto-carlos](https://github.com/state-space-models/cuthberto-carlos)
