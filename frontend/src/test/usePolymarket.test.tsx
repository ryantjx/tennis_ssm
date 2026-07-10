import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MatchPrediction } from "../types";
import {
  parsePolymarketTennisMarkets,
  predictionsForMatches,
  usePolymarket,
} from "../usePolymarket";
import { matchKey } from "../utils";

function match(overrides: Partial<MatchPrediction> = {}): MatchPrediction {
  return {
    id: "future-wimbledon-final",
    date: "2026-07-11",
    timestamp: 1288,
    player1: "Karolina Muchova",
    player2: "Linda Noskova",
    p_player1_win: 0.632,
    p_player2_win: 0.368,
    predicted_winner: "Karolina Muchova",
    actual_winner: null,
    correct: null,
    confidence: 0.632,
    log_score: null,
    player1_skill: 2.77,
    player2_skill: 1.54,
    player1_skill_sd: 0.82,
    player2_skill_sd: 0.82,
    tournament: "The Championships, Wimbledon",
    location: "WIMBLEDON",
    tier: "Grand Slam",
    surface: "Grass",
    round: "Final",
    is_future: true,
    ...overrides,
  };
}

function moneylineEvent() {
  return {
    id: "event-1",
    title: "Wimbledon WTA: Linda Noskova vs Karolina Muchova",
    slug: "wta-noskova-muchova-2026-07-11",
    eventDate: "2026-07-11T13:00:00Z",
    updatedAt: "2026-07-10T12:00:00Z",
    markets: [
      {
        id: "market-1",
        question: "Wimbledon WTA: Linda Noskova vs Karolina Muchova",
        slug: "moneyline",
        sportsMarketType: "moneyline",
        outcomes: JSON.stringify(["Linda Noskova", "Karolina Muchova"]),
        outcomePrices: JSON.stringify(["0.41", "0.59"]),
        active: true,
        closed: false,
        updatedAt: "2026-07-10T12:05:00Z",
        volume: "250000",
        liquidity: "50000",
      },
    ],
  };
}

describe("Polymarket tennis matching", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("parses tennis moneylines by canonical player pair", () => {
    const markets = parsePolymarketTennisMarkets([moneylineEvent()]);

    expect(Object.keys(markets)).toEqual(["karolina muchova|linda noskova"]);
    expect(markets["karolina muchova|linda noskova"][0]).toMatchObject({
      outcome1: "Linda Noskova",
      outcome2: "Karolina Muchova",
      price1: 0.41,
      price2: 0.59,
    });
  });

  it("orients market prices to the model player order", () => {
    const predictions = predictionsForMatches([match()], [moneylineEvent()]);

    expect(predictions[matchKey(match())]).toMatchObject({
      player1_market_name: "Karolina Muchova",
      player2_market_name: "Linda Noskova",
      player1_price: 0.59,
      player2_price: 0.41,
      matched_by: "live_canonical_player_pair",
    });
  });

  it("keeps markets separate when source-local match IDs repeat", () => {
    const first = match({
      id: "future-LS004",
      source: "wta_api",
      source_match_id: "LS004",
      source_tournament_id: "100",
    });
    const second = match({
      id: "future-LS004",
      player1: "Player C",
      player2: "Player D",
      predicted_winner: "Player C",
      tournament: "Second Open",
      source: "wta_api",
      source_match_id: "LS004",
      source_tournament_id: "200",
    });
    const secondEvent = {
      id: "event-2",
      title: "Second Open WTA: Player C vs Player D",
      slug: "wta-player-c-player-d-2026-07-11",
      eventDate: "2026-07-11T13:00:00Z",
      markets: [{
        id: "market-2",
        question: "Second Open WTA: Player C vs Player D",
        slug: "moneyline",
        sportsMarketType: "moneyline",
        outcomes: JSON.stringify(["Player C", "Player D"]),
        outcomePrices: JSON.stringify(["0.7", "0.3"]),
        active: true,
        closed: false,
      }],
    };

    const predictions = predictionsForMatches([first, second], [moneylineEvent(), secondEvent]);

    expect(matchKey(first)).not.toBe(matchKey(second));
    expect(predictions[matchKey(first)].event_id).toBe("event-1");
    expect(predictions[matchKey(second)].event_id).toBe("event-2");
  });

  it("falls back to deployment-time markets when the live request fails", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("network unavailable"))));
    const fallback = {
      source: "polymarket",
      event_url: "https://polymarket.com/event/fallback",
      player1_price: 0.6,
      player2_price: 0.4,
    };
    const matches = [match({ market: fallback })];
    const source = {
      dataUrl: "https://gamma-api.polymarket.com/events/keyset",
      tagSlug: "tennis",
    };

    const { result } = renderHook(() => usePolymarket(matches, source));

    await waitFor(() => expect(result.current.status).toBe("fallback"));
    expect(result.current.predictions[matchKey(matches[0])]).toEqual(fallback);
  });
});
