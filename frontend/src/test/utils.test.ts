import { describe, expect, it } from "vitest";
import { applyMatchFilters, confidenceLabel, formatPercent, matchKey } from "../utils";
import type { MatchPrediction } from "../types";

const baseMatch: MatchPrediction = {
  id: "m1",
  date: "2026-01-04",
  timestamp: 1100,
  player1: "Player A",
  player2: "Player B",
  p_player1_win: 0.62,
  p_player2_win: 0.38,
  predicted_winner: "Player A",
  actual_winner: "Player A",
  correct: true,
  confidence: 0.62,
  log_score: -0.48,
  player1_skill: 1.2,
  player2_skill: 0.2,
  player1_skill_sd: 0.4,
  player2_skill_sd: 0.5,
  tournament: "Test Open",
  location: "London",
  tier: "WTA500",
  surface: "Grass",
  round: "Final",
  is_future: false,
};

describe("prediction utilities", () => {
  it("formats percentages", () => {
    expect(formatPercent(0.631, 1)).toBe("63.1%");
  });

  it("labels confidence tiers", () => {
    expect(confidenceLabel(0.82)).toBe("High");
    expect(confidenceLabel(0.55)).toBe("Close");
  });

  it("filters by search and surface", () => {
    const matches = [
      baseMatch,
      { ...baseMatch, id: "m2", player1: "Player C", surface: "Hard", tournament: "City Cup" },
    ];

    const filtered = applyMatchFilters(matches, {
      query: "city",
    });

    expect(filtered).toHaveLength(1);
    expect(filtered[0].id).toBe("m2");
  });

  it("builds distinct keys for repeated source-local IDs", () => {
    const first = {
      ...baseMatch,
      id: "future-LS004",
      source: "wta_api",
      source_match_id: "LS004",
      tournament: "First Open",
    };
    const second = {
      ...first,
      tournament: "Second Open",
      player1: "Player C",
      player2: "Player D",
    };

    expect(matchKey(first)).not.toBe(matchKey(second));
  });
});
