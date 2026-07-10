import type { MatchFilters, MatchPrediction } from "./types";

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

export function matchKey(match: MatchPrediction): string {
  return [
    match.id,
    match.date,
    match.source,
    match.source_tournament_id,
    match.source_match_id,
    match.tournament,
    match.player1,
    match.player2,
  ].map(identityPart).join("::");
}

function identityPart(value: string | undefined): string {
  return (value ?? "").normalize("NFKC").trim().toLowerCase().replace(/\s+/g, " ");
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

export function applyMatchFilters(
  matches: MatchPrediction[],
  filters: MatchFilters,
): MatchPrediction[] {
  const query = filters.query.trim().toLowerCase();
  return matches.filter((match) => {
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
    return true;
  });
}
