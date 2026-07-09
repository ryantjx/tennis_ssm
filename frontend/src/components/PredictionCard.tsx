import type { CSSProperties } from "react";
import type { MatchPrediction } from "../types";
import {
  confidenceLabel,
  formatDate,
  formatPercent,
  formatSigned,
  probabilityForPredictedWinner,
} from "../utils";
import { MarketComparisonTable } from "./MarketComparisonTable";

interface PredictionCardProps {
  match: MatchPrediction;
  onOpen: (match: MatchPrediction) => void;
}

export function PredictionCard({ match, onOpen }: PredictionCardProps) {
  const predictedProbability = probabilityForPredictedWinner(match);
  const isFuture = !!match.is_future;
  const highlightedPlayer = match.actual_winner ?? match.predicted_winner;
  const status = match.match_status === "in_progress"
    ? "Live"
    : match.match_status === "suspended"
      ? "Suspended"
      : isFuture
        ? "Upcoming"
        : match.correct
          ? "Correct"
          : "Wrong";

  return (
    <article
      className={`prediction-card ${isFuture ? "prediction-card--future" : ""}`}
      tabIndex={0}
      onClick={() => onOpen(match)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen(match);
        }
      }}
      aria-label={`${match.player1} versus ${match.player2}`}
    >
      <div className="prediction-card__meta">
        <div className="prediction-card__meta-row">
          <strong>{formatDate(match.date)}</strong>
          <span className={`status-pill status-pill--${status.toLowerCase()}`}>{status}</span>
        </div>
        <div className="prediction-card__meta-row">
          <span className="eyebrow">{match.surface || "Unknown surface"} · {match.round || "Unknown round"}</span>
          <span className="prediction-card__location">{match.location && match.location !== "Unknown" ? match.location : ""}</span>
        </div>
        <span>{match.tournament}</span>
      </div>
      <div className="prediction-card__players">
        <PlayerLine name={match.player1} probability={match.p_player1_win} isPick={highlightedPlayer === match.player1} />
        <PlayerLine name={match.player2} probability={match.p_player2_win} isPick={highlightedPlayer === match.player2} />
      </div>
      {match.market ? <MarketComparisonTable match={match} compact /> : null}
      <div className="prediction-card__footer">
        <span>{confidenceLabel(match.confidence)} {formatPercent(predictedProbability, 1)}</span>
        <span>{formatSigned(match.player1_skill - match.player2_skill)} skill gap</span>
      </div>
    </article>
  );
}

function PlayerLine({ name, probability, isPick }: { name: string; probability: number; isPick: boolean }) {
  return (
    <div
      className={`player-line ${isPick ? "player-line--pick" : ""}`}
      style={{ "--probability": `${Math.max(0, Math.min(1, probability)) * 100}%` } as CSSProperties}
    >
      <span className="player-line__fill" aria-hidden="true" />
      <strong>{name}</strong>
      <span>{formatPercent(probability, 1)}</span>
    </div>
  );
}
