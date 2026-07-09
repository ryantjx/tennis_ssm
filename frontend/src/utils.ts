import type { MatchFilters, MatchPrediction, SortMode } from "./types";

export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatSigned(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function formatDate(date: string): string {
  return new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${date}T12:00:00Z`));
}

export function matchId(match: MatchPrediction): string {
  return match.id ?? `${match.date}-${match.player1}-${match.player2}`;
}

export function probabilityForPredictedWinner(match: MatchPrediction): number {
  return match.predicted_winner === match.player1
    ? match.p_player1_win
    : match.p_player2_win;
}

export function confidenceLabel(confidence: number): string {
  if (confidence >= 0.8) return "High";
  if (confidence >= 0.68) return "Likely";
  if (confidence >= 0.58) return "Lean";
  return "Close";
}

export function surfaceOptions(matches: MatchPrediction[]): string[] {
  return Array.from(
    new Set(matches.map((match) => match.surface).filter((surface) => surface && surface !== "Unknown")),
  ).sort();
}

export function applyMatchFilters(
  matches: MatchPrediction[],
  filters: MatchFilters,
): MatchPrediction[] {
  const query = filters.query.trim().toLowerCase();
  const filtered = matches.filter((match) => {
    if (query) {
      const haystack = [
        match.player1,
        match.player2,
        match.tournament,
        match.location,
        match.round,
      ].join(" ").toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    if (filters.surface && match.surface !== filters.surface) return false;
    return true;
  });

  return filtered.sort((left, right) => compareMatches(left, right, filters.sort));
}

export function compareMatches(left: MatchPrediction, right: MatchPrediction, sort: SortMode): number {
  switch (sort) {
    case "date-asc":
      return left.timestamp - right.timestamp;
    case "date-desc":
      return right.timestamp - left.timestamp;
    case "confidence-asc":
      return left.confidence - right.confidence;
    case "confidence-desc":
      return right.confidence - left.confidence;
    case "edge-desc":
      return Math.abs(right.p_player1_win - 0.5) - Math.abs(left.p_player1_win - 0.5);
    case "log-score-asc":
      return scoreValue(left.log_score) - scoreValue(right.log_score);
    case "log-score-desc":
      return scoreValue(right.log_score) - scoreValue(left.log_score);
    default:
      return 0;
  }
}

function scoreValue(value: number | null): number {
  return value ?? -999;
}
