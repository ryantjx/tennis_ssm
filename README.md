# State-Space Models for Tennis

State-space models for tennis predictions.

## Model

Following Duffield, Power and Rimella (2024), the model for tennis can be defined as follows,

$$p(x_0) \sim \mathcal{N}(\mu_0, \Sigma_0)$$

$$p(x_t | x_{t-1}) \sim \mathcal{N}(\tau_d \cdot \Delta t, Q_k)$$

$$G_k (y_k \mid x^{(i)}, x^{(j)}) = \begin{cases} \sigma(\frac{x^{(i)} - x^{(j)}}{s_d}) & \text{if } y_k = x^{(i)} \\ 1 - \sigma(\frac{x^{(i)} - x^{(j)}}{s_d}) & \text{if } y_k = x^{(j)} \end{cases}$$

where 

- $y_k$ is the outcome of match $k$ between players $i$ and $j$.
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function. As $\sigma(x) \to \infty$, $G_k$ converges to a uniform distribution over the two players.
- $\tau_d \in \mathbb{R}^+$ is a rate parameter that controls evolution of skill for player
- $s_d \in \mathbb{R}^+$ is a scale parameter

The parameters to be estimated are $\mu_0, \Sigma_0, \tau_d, s_d$.

## Algorithm

We use the Gaussian Moments filter from `cuthbert` to perform inference, approximating the posterior distribution as a Gaussian with Kalman Filter updates. Player skills are also assumed independent given observations up to time $t$, with each player's posterior represented using a Cholesky decomposition:

$$p(x^{(i)}_t \mid y_{1:t}) \approx \mathcal{N}(\mu^{(i)}_t, \Sigma^{(i)}_t)$$

$$p(x^{(i)}_t, x^{(j)}_t | y_{1:t}) \propto p(x^{(i)}_t \mid y_{1:t}) p(x^{(j)}_t \mid y_{1:t})$$

## Data

Data is collected http://www.tennis-data.co.uk/alldata.php.

## Results

The model is trained on WTA matches from 2022-2024, selects parameters on 2025 matches, and evaluates the selected model on 2026 matches.

| Metric | Value |
|---|---|
| Training matches | 7,486 |
| Test matches | 1,422 |
| Accuracy | 63.1% |
| Avg log-score | -0.6378 |
| Uniform baseline | -0.6931 |

Trained parameters: τ=0.029, s=1.76, init_var=1.71

### Prediction outputs

The canonical model output is generated at `outputs/predictions.json`. The
deployment workflow copies that file to `frontend/public/data/predictions.json`
before building the frontend; the copied file is ignored because it is generated
runtime data.

The deployment path reads `tau`, `s`, and `init_var` from
`outputs/tennis_factorial_state.json`, then reruns the filter over all completed
WTA matches from the model origin through the latest available results. The
latest filtered state is synchronized to the newest completed match timestamp
before future fixture predictions are generated. The exported JSON records the
training/test/prediction windows, saved parameters, validation metrics,
historical predictions, player rankings from the latest filtered state, and
matched WTA future fixture predictions.

Daily generation writes the newest deployable data to
`outputs/latest/predictions.json` and `outputs/latest/results.json`, and stores a
dated snapshot under `outputs/daily/YYYY-MM-DD/`. GitHub Pages deploys the React
app shell separately; in production the app fetches the latest JSON directly
from `outputs/latest` on the `main` branch, so daily prediction updates do not
require a Pages redeploy.

Future predictions are loaded from the WTA fixture API via
`src/data/fixtures_womens.py` and filtered to players known by the trained model.
Synthetic top-player matchups are disabled in the main workflow.

### Frontend

Predictions are deployed to GitHub Pages at **https://ryantjx.github.io/tennis_ssm/**.

The frontend is a Vite React app in `frontend/` inspired by the
[cuthberto-carlos](https://github.com/state-space-models/cuthberto-carlos/tree/main/frontend)
interaction model, adapted for general tennis data. It features:
- **Upcoming predictions** loaded from matched WTA future fixtures
- **Completed forecast archive** with prediction quality and match detail drawer
- **Committed results reference** at `frontend/public/data/results.json`
- **Search** across predictions and player rankings
- **Player rankings** with all filtered skill ratings and uncertainty

Local frontend commands:

```bash
cd frontend
npm install
npm test
npm run build
```

## Extensions

Current model only models outcome of {win, loss} and can model {draw} with a few modifications. Football has {draw} and no upper bound on scores (though we can truncate the likelihood model to a reasonable number), so then bivariate poisson works. A solution to this would be to use Bernoulli for likelihood for a set and then use a exhaustive summation for the combination of sets. (For bo3, it would be 2-1, 2-0, 1-2, 0-2). Or an ordered probit model where we just model likelihood of score difference (e.g. 2-0 is +2. then combination would be +2, +1, -1, -2).

## References

- Samuel Duffield, Samuel Power, Lorenzo Rimella, A state-space perspective on modelling and inference for online skill rating, Journal of the Royal Statistical Society Series C: Applied Statistics, Volume 73, Issue 5, November 2024, Pages 1262–1282, https://doi.org/10.1093/jrsssc/qlae035
  - [SamDuffield/abile](https://github.com/SamDuffield/abile)
  - data: https://github.com/SamDuffield/abile/blob/main/datasets/tennis.py
  - train: https://github.com/SamDuffield/abile/blob/main/simulations/tennis_train.py
  - model: https://github.com/SamDuffield/abile/blob/main/abile/models/extended_kalman.py
- [state-space-models/cuthberto-carlos](https://github.com/state-space-models/cuthberto-carlos)
