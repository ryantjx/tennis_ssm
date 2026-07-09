import type { MatchPrediction } from "../types";
import {
  formatDate,
  formatPercent,
  formatSigned,
  probabilityForPredictedWinner,
} from "../utils";
import { MarketComparisonTable } from "./MarketComparisonTable";

interface MatchDetailDrawerProps {
  match: MatchPrediction | null;
  onClose: () => void;
}

export function MatchDetailDrawer({ match, onClose }: MatchDetailDrawerProps) {
  if (!match) return null;

  const predictedProbability = probabilityForPredictedWinner(match);

  return (
    <div className="drawer-backdrop" role="presentation" onClick={onClose}>
      <aside
        className="match-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="drawer-title"
        onClick={(event) => event.stopPropagation()}
      >
        <button className="drawer-close" type="button" onClick={onClose} aria-label="Close match details">X</button>
        <span className="eyebrow">{match.tournament} · {match.surface}</span>
        <h2 id="drawer-title">{match.player1} vs {match.player2}</h2>
        <p>{formatDate(match.date)} · {match.location} · {match.round}</p>

        <div className="drawer-pick" aria-label="Model pick">
          <span>Model pick</span>
          <strong>{match.predicted_winner}</strong>
          <b>{formatPercent(predictedProbability, 1)}</b>
        </div>

        <div className="drawer-summary">
          <section aria-label="Prediction">
            <h3>Prediction</h3>
            <dl>
              <div>
                <dt>
                  <span>{match.player1}{match.player1_rank ? <em>Rank #{match.player1_rank}</em> : null}</span>
                  <small>{formatSigned(match.player1_skill, 2)} ±{match.player1_skill_sd.toFixed(2)}</small>
                </dt>
                <dd>{formatPercent(match.p_player1_win, 1)}</dd>
              </div>
              <div>
                <dt>
                  <span>{match.player2}{match.player2_rank ? <em>Rank #{match.player2_rank}</em> : null}</span>
                  <small>{formatSigned(match.player2_skill, 2)} ±{match.player2_skill_sd.toFixed(2)}</small>
                </dt>
                <dd>{formatPercent(match.p_player2_win, 1)}</dd>
              </div>
            </dl>
          </section>
          <section aria-label="Outcome">
            <h3>Outcome</h3>
            <dl>
              <div>
                <dt>Winner</dt>
                <dd>{match.actual_winner ?? "Not yet played"}</dd>
              </div>
              <div>
                <dt>Result</dt>
                <dd>{match.is_future ? "Upcoming" : match.correct ? "Correct" : "Wrong"}</dd>
              </div>
              <div>
                <dt>Log score</dt>
                <dd>{match.log_score === null ? "Pending" : match.log_score.toFixed(4)}</dd>
              </div>
            </dl>
          </section>
        </div>

        {match.market ? (
          <section className="drawer-section" aria-label="Market comparison">
            <MarketComparisonTable match={match} />
          </section>
        ) : null}
      </aside>
    </div>
  );
}
