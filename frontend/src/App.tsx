import { useEffect, useMemo, useState } from "react";
import { CompletedResults } from "./components/CompletedResults";
import { FilterControls } from "./components/FilterControls";
import { MatchDetailDrawer } from "./components/MatchDetailDrawer";
import { PlayerRankings } from "./components/PlayerRankings";
import { UpcomingMatches } from "./components/UpcomingMatches";
import type { MatchFilters, MatchPrediction, PredictionPayload, ResultsPayload } from "./types";
import { applyMatchFilters, formatDate, formatPercent } from "./utils";

const defaultFilters: MatchFilters = {
  query: "",
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
  const filteredPlayers = useMemo(() => {
    const query = filters.query.trim().toLowerCase();
    if (!data) return [];
    if (!query) return data.top_players;
    return data.top_players.filter((player) => player.name.toLowerCase().includes(query));
  }, [data, filters.query]);

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
  const trainWindow = data.data_windows?.train_display_start && data.data_windows?.train_display_end
    ? `${data.data_windows.train_display_start} to ${data.data_windows.train_display_end}`
    : `${data.data_windows?.train_start ?? "unknown"} to ${data.data_windows?.train_end ?? "unknown"}`;
  const tuningWindow = evaluatedTuningWindow(data);

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
              Parameters are selected on 2025 test log score, then the selected model is used for 2026 onward predictions and future fixtures.
            </p>
          </div>
          <dl className="summary-card" aria-label="Overall model results">
            <div className="summary-card__stats">
              <div>
                <dt>Accuracy</dt>
                <dd>{formatPercent(data.metrics.accuracy, 1)}</dd>
                <span>Correct picks across evaluated test and post-selection results</span>
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
          resultCount={filteredMatches.length}
          totalCount={allMatches.length}
        />

        <UpcomingMatches matches={upcoming} onOpen={setSelectedMatch} fixtureStatus={data.fixture_status} />
        <CompletedResults
          matches={completed}
          results={results.results}
          onOpen={setSelectedMatch}
          resultWindow={results.data_window}
        />
        <PlayerRankings players={filteredPlayers} />
        <section className="methodology-section" id="methodology" aria-labelledby="methodology-title">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Model notes</span>
              <h2 id="methodology-title">Methodology</h2>
            </div>
          </div>
          <div className="methodology-grid">
            <article className="methodology-brief">
              <h3>Model brief</h3>
              <p>
                A Gaussian factorial state-space model tracks one latent skill per WTA player. Skills evolve through a random walk, and win probability comes from the skill difference through a logistic observation model. Filtering uses Gaussian moment updates over historical match outcomes:
              </p>
              <div className="equation-list" aria-label="Model equations">
                <MathEquation label="Initial skill prior" mathml="<math display='block'><mrow><mi>p</mi><mo>(</mo><msub><mi>x</mi><mn>0</mn></msub><mo>)</mo><mo>~</mo><mi>N</mi><mo>(</mo><msub><mi>μ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>Σ</mi><mn>0</mn></msub><mo>)</mo></mrow></math>" />
                <MathEquation label="Skill evolution" mathml="<math display='block'><mrow><mi>p</mi><mo>(</mo><msub><mi>x</mi><mi>t</mi></msub><mo>|</mo><msub><mi>x</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub><mo>)</mo><mo>~</mo><mi>N</mi><mo>(</mo><msub><mi>τ</mi><mi>d</mi></msub><mo>Δ</mo><mi>t</mi><mo>,</mo><msub><mi>Q</mi><mi>k</mi></msub><mo>)</mo></mrow></math>" />
                <MathEquation label="Match observation probability" mathml="<math display='block'><mrow><msub><mi>G</mi><mi>k</mi></msub><mo>(</mo><msub><mi>y</mi><mi>k</mi></msub><mo>|</mo><msup><mi>x</mi><mi>i</mi></msup><mo>,</mo><msup><mi>x</mi><mi>j</mi></msup><mo>)</mo><mo>=</mo><mi>σ</mi><mo>(</mo><mfrac><mrow><msup><mi>x</mi><mi>i</mi></msup><mo>-</mo><msup><mi>x</mi><mi>j</mi></msup></mrow><msub><mi>s</mi><mi>d</mi></msub></mfrac><mo>)</mo></mrow></math>" />
                <MathEquation label="Gaussian filtered posterior" mathml="<math display='block'><mrow><mi>p</mi><mo>(</mo><msubsup><mi>x</mi><mi>t</mi><mi>i</mi></msubsup><mo>|</mo><msub><mi>y</mi><mrow><mn>1</mn><mo>:</mo><mi>t</mi></mrow></msub><mo>)</mo><mo>≈</mo><mi>N</mi><mo>(</mo><msubsup><mi>μ</mi><mi>t</mi><mi>i</mi></msubsup><mo>,</mo><msubsup><mi>Σ</mi><mi>t</mi><mi>i</mi></msubsup><mo>)</mo></mrow></math>" />
              </div>
            </article>
            <aside className="methodology-params-panel">
              <h3>Model parameters <span>Latest run: {generatedAt.toLocaleString()}</span></h3>
              <ul className="methodology-params" aria-label="Model parameters">
                <li><strong><MathInline label="tau sub d" mathml="<math><msub><mi>τ</mi><mi>d</mi></msub></math>" />:</strong> {data.model_params.tau.toFixed(6)}</li>
                <li><strong><MathInline label="s sub d" mathml="<math><msub><mi>s</mi><mi>d</mi></msub></math>" />:</strong> {data.model_params.s.toFixed(6)}</li>
                <li><strong>Initial variance <MathInline label="Sigma sub zero" mathml="<math><msub><mi>Σ</mi><mn>0</mn></msub></math>" />:</strong> {data.model_params.init_var.toFixed(6)}</li>
                <li><strong>Initial mean <MathInline label="mu sub zero" mathml="<math><msub><mi>μ</mi><mn>0</mn></msub></math>" />:</strong> 0.000000</li>
                <li><strong>Train:</strong> {trainWindow}</li>
                <li><strong>Test:</strong> {tuningWindow}</li>
              </ul>
            </aside>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div>
          <strong>WTA Match Prediction Model</strong>
          <span>Gaussian factorial state-space model for tennis match outcomes.</span>
        </div>
        <p>
          Completed results source: {results.source}. Latest result window starts {results.data_window ? formatDate(results.data_window.start) : "unknown"}.
        </p>
        <p>
          Repository: <a href="https://github.com/ryantjx/tennis_ssm" target="_blank" rel="noreferrer">ryantjx/tennis_ssm</a>
        </p>
      </footer>

      <MatchDetailDrawer match={selectedMatch} onClose={() => setSelectedMatch(null)} />
    </>
  );
}

function MathEquation({ label, mathml }: { label: string; mathml: string }) {
  return (
    <div
      className="math-equation"
      role="math"
      aria-label={label}
      dangerouslySetInnerHTML={{ __html: mathml }}
    />
  );
}

function MathInline({ label, mathml }: { label: string; mathml: string }) {
  return (
    <span
      className="math-inline"
      role="math"
      aria-label={label}
      dangerouslySetInnerHTML={{ __html: mathml }}
    />
  );
}

function evaluatedTuningWindow(data: PredictionPayload): string {
  const start = data.data_windows?.test_match_start;
  const end = data.data_windows?.test_match_end;
  if (start && end) return `${start} to ${end}`;
  if (data.data_windows?.test_display_start && data.data_windows?.test_display_end) {
    return `${data.data_windows.test_display_start} to ${data.data_windows.test_display_end}`;
  }
  return "unknown";
}

export default App;
