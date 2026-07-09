import type { MatchPrediction } from "../types";
import { confidenceLabel, formatDate, formatPercent, formatSigned, probabilityForPredictedWinner } from "../utils";

interface PredictionListRowProps {
  match: MatchPrediction;
  onOpen: (match: MatchPrediction) => void;
}

export function PredictionListRow({ match, onOpen }: PredictionListRowProps) {
  const highlightedPlayer = match.actual_winner ?? match.predicted_winner;
  const status: string | null = match.match_status === "in_progress"
    ? "Live"
    : match.match_status === "suspended"
      ? "Suspended"
      : match.is_future
        ? null
        : match.correct
          ? "Correct"
          : "Wrong";
  return (
    <article
      className="prediction-row"
      tabIndex={0}
      onClick={() => onOpen(match)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen(match);
        }
      }}
    >
      <div>
        <span className="eyebrow">{formatDate(match.date)}</span>
        <strong>{match.tournament}</strong>
        <span>{match.surface} · {match.round}</span>
      </div>
      <div className="prediction-row__fixture">
        <span className={highlightedPlayer === match.player1 ? "row-highlight" : ""}>
          <b>{match.player1}{match.player1_rank ? <em>Rank #{match.player1_rank}</em> : null}</b>
          <small>{formatSigned(match.player1_skill, 2)} ±{match.player1_skill_sd.toFixed(2)}</small>
        </span>
        <strong>{formatPercent(match.p_player1_win, 0)} - {formatPercent(match.p_player2_win, 0)}</strong>
        <span className={highlightedPlayer === match.player2 ? "row-highlight" : ""}>
          <b>{match.player2}{match.player2_rank ? <em>Rank #{match.player2_rank}</em> : null}</b>
          <small>{formatSigned(match.player2_skill, 2)} ±{match.player2_skill_sd.toFixed(2)}</small>
        </span>
      </div>
      {status
        ? <span className={`status-pill status-pill--${status.toLowerCase()}`}>{status}</span>
        : <span aria-hidden="true" />}
      <span>{confidenceLabel(match.confidence)} {formatPercent(probabilityForPredictedWinner(match), 1)}</span>
    </article>
  );
}
