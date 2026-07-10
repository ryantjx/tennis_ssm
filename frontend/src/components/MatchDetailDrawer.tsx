import { useEffect, useRef } from "react";
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
  const drawerRef = useRef<HTMLElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!match) return;

    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !drawerRef.current) return;

      const focusable = Array.from(drawerRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ));
      if (!focusable.length) {
        event.preventDefault();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && (document.activeElement === first || !drawerRef.current.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !drawerRef.current.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    }

    const drawer = drawerRef.current;
    drawer?.addEventListener("keydown", handleKeyDown);
    return () => {
      drawer?.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousBodyOverflow;
      previouslyFocused?.focus();
    };
  }, [match, onClose]);

  if (!match) return null;

  const predictedProbability = probabilityForPredictedWinner(match);
  const predictionResult = match.is_future ? "Upcoming" : match.correct ? "Correct" : "Wrong";

  return (
    <div className="drawer-backdrop" role="presentation" onClick={onClose}>
      <aside
        ref={drawerRef}
        className="match-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="drawer-title"
        onClick={(event) => event.stopPropagation()}
      >
        <button ref={closeRef} className="drawer-close" type="button" onClick={onClose} aria-label="Close match details">×</button>
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
                <dt>Prediction</dt>
                <dd className={`outcome-pill outcome-pill--${predictionResult.toLowerCase()}`}>{predictionResult}</dd>
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
