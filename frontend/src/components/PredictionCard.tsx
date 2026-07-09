import type { CSSProperties } from "react";
import type { MatchPrediction } from "../types";
import {
  formatDate,
  formatPercent,
  formatSigned,
} from "../utils";
import { MarketComparisonTable } from "./MarketComparisonTable";

interface PredictionCardProps {
  match: MatchPrediction;
  onOpen: (match: MatchPrediction) => void;
}

export function PredictionCard({ match, onOpen }: PredictionCardProps) {
  const isFuture = !!match.is_future;
  const highlightedPlayer = match.actual_winner ?? match.predicted_winner;
  const status: string | null = match.match_status === "in_progress"
    ? "Live"
    : match.match_status === "suspended"
      ? "Suspended"
      : isFuture
        ? null
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
          {status ? <span className={`status-pill status-pill--${status.toLowerCase()}`}>{status}</span> : null}
        </div>
        <div className="prediction-card__meta-row">
          <span className="eyebrow">{match.surface || "Unknown surface"} · {match.round || "Unknown round"}</span>
          <span className="prediction-card__location">{match.location && match.location !== "Unknown" ? match.location : ""}</span>
        </div>
        <span>{match.tournament}</span>
      </div>
      <div className="prediction-card__players">
        <PlayerLine
          name={match.player1}
          rank={match.player1_rank}
          probability={match.p_player1_win}
          skill={match.player1_skill}
          skillSd={match.player1_skill_sd}
          isPick={highlightedPlayer === match.player1}
        />
        <PlayerLine
          name={match.player2}
          rank={match.player2_rank}
          probability={match.p_player2_win}
          skill={match.player2_skill}
          skillSd={match.player2_skill_sd}
          isPick={highlightedPlayer === match.player2}
        />
      </div>
      {match.market ? <MarketComparisonTable match={match} compact /> : null}
    </article>
  );
}

function PlayerLine({
  name,
  rank,
  probability,
  skill,
  skillSd,
  isPick,
}: {
  name: string;
  rank?: number | null;
  probability: number;
  skill: number;
  skillSd: number;
  isPick: boolean;
}) {
  return (
    <div
      className={`player-line ${isPick ? "player-line--pick" : ""}`}
      style={{ "--probability": `${Math.max(0, Math.min(1, probability)) * 100}%` } as CSSProperties}
    >
      <span className="player-line__fill" aria-hidden="true" />
      <span className="player-line__identity">
        <strong>
          <span>{name}</span>
          {rank ? <em>Rank #{rank}</em> : null}
        </strong>
        <small>{formatSigned(skill, 2)} ±{skillSd.toFixed(2)}</small>
      </span>
      <b>{formatPercent(probability, 1)}</b>
    </div>
  );
}
