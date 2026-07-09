import { useEffect, useMemo, useState } from "react";
import { CompletedResults } from "./components/CompletedResults";
import { FilterControls } from "./components/FilterControls";
import { MatchDetailDrawer } from "./components/MatchDetailDrawer";
import { PlayerRankings } from "./components/PlayerRankings";
import { UpcomingMatches } from "./components/UpcomingMatches";
import type { MatchFilters, MatchPrediction, PredictionPayload, ResultsPayload } from "./types";
import { applyMatchFilters, formatDate, formatPercent, surfaceOptions } from "./utils";

const defaultFilters: MatchFilters = {
  query: "",
  surface: "",
  sort: "date-desc",
};

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

function App() {
  const [data, setData] = useState<PredictionPayload | null>(null);
  const [results, setResults] = useState<ResultsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<MatchPrediction | null>(null);
  const [filters, setFilters] = useState<MatchFilters>(defaultFilters);

  useEffect(() => {
    Promise.all([
      fetchJson<PredictionPayload>("data/predictions.json"),
      fetchJson<ResultsPayload>("data/results.json"),
    ])
      .then(([predictionPayload, resultPayload]) => {
        setData(predictionPayload);
        setResults(resultPayload);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
      });
  }, []);

  const allMatches = useMemo(() => {
    if (!data) return [];
    return [...(data.future_matches ?? []), ...(data.matches ?? [])];
  }, [data]);

  const filteredMatches = useMemo(
    () => applyMatchFilters(allMatches, filters),
    [allMatches, filters],
  );
  const surfaces = useMemo(() => surfaceOptions(allMatches), [allMatches]);

  if (error) {
    return (
      <main className="load-state">
        <strong>Prediction data could not be loaded.</strong>
        <span>{error}</span>
      </main>
    );
  }

  if (!data || !results) {
    return (
      <main className="load-state">
        <strong>Loading tennis forecasts</strong>
        <span>Reading model predictions and completed results.</span>
      </main>
    );
  }

  const generatedAt = new Date(data.generated_at);
  const logDelta = data.metrics.avg_log_score - data.metrics.uniform_baseline;
  const upcoming = filteredMatches.filter((match) => match.is_future);
  const completed = filteredMatches.filter((match) => !match.is_future);
  const matchedMarkets = data.market_status?.matched_model_matches ?? allMatches.filter((match) => match.market).length;

  return (
    <>
      <header className="site-header">
        <a className="site-brand" href="#top" aria-label="WTA Match Prediction Model home">
          <span className="site-brand__mark" aria-hidden="true" />
          <span>WTA Match Prediction Model</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#upcoming">Upcoming</a>
          <a href="#completed">Results</a>
          <a href="#rankings">Rankings</a>
          <a href="#methodology">Methodology</a>
        </nav>
        <div className="header-snapshot">
          <span>Latest model run</span>
          <strong>{generatedAt.toLocaleDateString()}</strong>
          <small>{generatedAt.toLocaleTimeString()}</small>
        </div>
      </header>

      <main id="top">
        <section className="dashboard" aria-labelledby="dashboard-title">
          <div className="dashboard__copy">
            <span className="eyebrow">WTA Match Prediction Model</span>
            <h1 id="dashboard-title">Match forecasts and completed results</h1>
            <p>
              Predictions are optimized against the 2026 test log score and future fixtures are sourced from the WTA schedule when players match the trained model.
            </p>
          </div>
          <dl className="summary-card" aria-label="Overall model results">
            <div className="summary-card__stats">
              <div>
                <dt>Accuracy</dt>
                <dd>{formatPercent(data.metrics.accuracy, 1)}</dd>
              </div>
              <div>
                <dt>Log score</dt>
                <dd>{data.metrics.avg_log_score.toFixed(4)}</dd>
                <span>{logDelta >= 0 ? "+" : ""}{logDelta.toFixed(4)} vs uniform</span>
              </div>
            </div>
          </dl>
        </section>

        <FilterControls
          filters={filters}
          onChange={setFilters}
          surfaces={surfaces}
        />

        <UpcomingMatches matches={upcoming} onOpen={setSelectedMatch} fixtureStatus={data.fixture_status} />
        <CompletedResults
          matches={completed}
          results={results.results}
          onOpen={setSelectedMatch}
          resultWindow={results.data_window}
        />
        <PlayerRankings players={data.top_players} />
        <section className="methodology-section" id="methodology" aria-labelledby="methodology-title">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Model notes</span>
              <h2 id="methodology-title">Methodology</h2>
            </div>
            <p>Latest run: {generatedAt.toLocaleString()}</p>
          </div>
          <div className="methodology-grid">
            <article>
              <h3>State-space model</h3>
              <p>
                The model is a Gaussian factorial state-space model with one latent skill per WTA player. Player skill follows a random walk over time, and match win probability is computed from the skill difference with a logistic observation model.
              </p>
            </article>
            <article>
              <h3>Training and objective</h3>
              <p>
                Historical WTA matches are filtered through the model, then smoothing parameters are selected by maximizing the 2026 test-set average log score. Future predictions use WTA fixtures only when both players resolve to trained model players.
              </p>
            </article>
            <article>
              <h3>Market comparison</h3>
              <p>
                Polymarket tennis moneylines are optional live inputs. When available, markets are matched by normalized singles player names and shown beside model-implied probabilities.
              </p>
            </article>
            <h3 className="methodology-params__title">Optimized model parameters</h3>
            <dl className="methodology-params" aria-label="Optimized model parameters">
              <div>
                <dt>Tau</dt>
                <dd>{data.model_params.tau.toFixed(6)}</dd>
              </div>
              <div>
                <dt>S</dt>
                <dd>{data.model_params.s.toFixed(6)}</dd>
              </div>
              <div>
                <dt>Init var</dt>
                <dd>{data.model_params.init_var.toFixed(6)}</dd>
              </div>
              <div>
                <dt>Objective</dt>
                <dd>{data.optimization?.objective ?? "Test log score optimization"}</dd>
              </div>
              <div>
                <dt>Train window</dt>
                <dd>{data.data_windows?.train_start ?? "unknown"} to {data.data_windows?.train_end ?? "unknown"}</dd>
              </div>
              <div>
                <dt>Test window</dt>
                <dd>{data.data_windows?.test_start ?? "unknown"} to {data.data_windows?.test_end ?? "unknown"}</dd>
              </div>
              <div>
                <dt>Market lines</dt>
                <dd>{matchedMarkets}</dd>
              </div>
              <div>
                <dt>Future fixtures</dt>
                <dd>{data.future_matches.length}</dd>
              </div>
            </dl>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div>
          <strong>WTA Match Prediction Model</strong>
          <span>Gaussian factorial state-space model for tennis match outcomes.</span>
        </div>
        <p>
          Parameters: tau {data.model_params.tau.toFixed(4)}, s {data.model_params.s.toFixed(4)}, init var {data.model_params.init_var.toFixed(4)}.
        </p>
        <p>
          Completed results source: {results.source}. Latest result window starts {results.data_window ? formatDate(results.data_window.start) : "unknown"}.
        </p>
      </footer>

      <MatchDetailDrawer match={selectedMatch} onClose={() => setSelectedMatch(null)} />
    </>
  );
}

export default App;
