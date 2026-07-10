import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { PredictionPayload, ResultsPayload } from "../types";

const predictions: PredictionPayload = {
  generated_at: "2026-07-09T12:00:00Z",
  data_windows: {
    train_start: "2021-12-31",
    train_end: "2024-12-31",
    train_display_start: "2022-01-01",
    train_display_end: "2024-12-31",
    test_start: "2024-12-31",
    test_end: "2025-12-31",
    test_match_start: "2025-01-01",
    test_match_end: "2025-12-31",
    prediction_display_start: "2026-01-01",
    prediction_display_end: "2026-12-31",
    upcoming_match_start: "2026-01-04",
    upcoming_match_end: "2026-06-27",
  },
  model_params: { tau: 0.1, s: 1.0, init_var: 1.0 },
  metrics: {
    n_test_matches: 1,
    n_future_matches: 1,
    accuracy: 1,
    avg_log_score: -0.4,
    uniform_baseline: -0.6931,
  },
  market_status: {
    source: "polymarket",
    loaded_events: 1,
    loaded_moneylines: 1,
    matched_model_matches: 1,
  },
  top_players: [
    { rank: 1, name: "Player A", skill: 1.2, variance: 0.2 },
    { rank: 2, name: "Player Negative", skill: -0.8, variance: 0.3 },
  ],
  matches: [
    {
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
    },
  ],
  future_matches: [
    {
      id: "f1",
      date: "2026-07-09",
      timestamp: 1286,
      player1: "Player C",
      player2: "Player D",
      p_player1_win: 0.55,
      p_player2_win: 0.45,
      predicted_winner: "Player C",
      actual_winner: null,
      correct: null,
      confidence: 0.55,
      log_score: null,
      player1_skill: 0.8,
      player2_skill: 0.5,
      player1_skill_sd: 0.5,
      player2_skill_sd: 0.5,
      player1_rank: 12,
      player2_rank: 18,
      player1_latest_skill: 0.8,
      player2_latest_skill: 0.5,
      player1_latest_variance: 0.3,
      player2_latest_variance: 0.4,
      market: {
        source: "polymarket",
        event_url: "https://polymarket.com/event/test",
        player1_price: 0.52,
        player2_price: 0.48,
        player1_edge: 0.03,
        player2_edge: -0.03,
      },
      tournament: "Future Open",
      location: "Paris",
      tier: "WTA250",
      surface: "Clay",
      round: "Semifinals",
      is_future: true,
    },
  ],
};

const results: ResultsPayload = {
  generated_at: "2026-07-09T12:00:00Z",
  source: "test",
  data_window: { start: "2025-12-31", end: "2027-01-01" },
  results: [
    {
      id: "r1",
      date: "2026-01-04",
      winner: "Player A",
      loser: "Player B",
      player1: "Player A",
      player2: "Player B",
      actual_winner: "Player A",
      tournament: "Test Open",
      location: "London",
      tier: "WTA500",
      surface: "Grass",
      round: "Final",
    },
  ],
};

describe("App", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn((url: string) => {
      const body = url.includes("results") ? results : predictions;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(body),
      });
    }));
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders upcoming predictions, completed results, and rankings", async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText("Upcoming predictions")).toBeInTheDocument());
    expect(screen.queryByText("Test matches")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Surface")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Sort")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "WTA Match Prediction State-Space Model" })).toBeInTheDocument();
    expect(screen.getAllByText("WTA Match Prediction State-Space Model")).toHaveLength(1);
    expect(screen.getByText(/A Gaussian factorial state-space model estimates WTA player skill/)).toBeInTheDocument();
    expect(screen.getByText("Accuracy")).toBeInTheDocument();
    expect(screen.getByText("1 / 1 correct predictions")).toBeInTheDocument();
    expect(screen.getByText("Log score")).toBeInTheDocument();
    expect(screen.getAllByText(/Future Open/).length).toBeGreaterThan(0);
    expect(screen.queryByText("Current WTA matches")).not.toBeInTheDocument();
    expect(screen.getByText("Completed results")).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Player 1" })).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Player 2" })).not.toBeInTheDocument();
    expect(screen.getByText("Player rankings")).toBeInTheDocument();
    expect(screen.getByText("Player Negative")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Methodology" })).toBeInTheDocument();
    expect(screen.getByText("Model parameters")).toBeInTheDocument();
    expect(screen.getByText("2022-01-01 to 2024-12-31")).toBeInTheDocument();
    expect(screen.getByText("2025-01-01 to 2025-12-31")).toBeInTheDocument();
    expect(screen.queryByText("2026-01-04 to 2026-06-27")).not.toBeInTheDocument();
    expect(screen.getByText("0.000000")).toBeInTheDocument();
    expect(screen.queryByText(/Parameters: tau/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Objective:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Market lines:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Future fixtures:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Prediction window:/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Model equations")).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "ryantjx/tennis_ssm" })[0]).toHaveAttribute(
      "href",
      "https://github.com/ryantjx/tennis_ssm",
    );
    expect(screen.getAllByText("Model vs Polymarket").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("columnheader", { name: "Market" }).length).toBeGreaterThan(0);
  });

  it("filters matches and rankings with the global search", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(screen.getByText("Upcoming predictions")).toBeInTheDocument());
    await user.type(screen.getByLabelText("Search"), "Future");
    expect(screen.getByText("Future Open")).toBeInTheDocument();
    expect(screen.queryByText("Test Open")).not.toBeInTheDocument();

    await user.clear(screen.getByLabelText("Search"));
    await user.type(screen.getByLabelText("Search"), "negative");
    expect(screen.getByText("Player Negative")).toBeInTheDocument();
    expect(screen.queryByText("Player A")).not.toBeInTheDocument();
    expect(screen.queryByText("Future Open")).not.toBeInTheDocument();
  });

  it("shows player ranks without repeated skill blocks in the match drawer", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(screen.getByText("Upcoming predictions")).toBeInTheDocument());
    const trigger = screen.getAllByLabelText("Player C versus Player D")[0];
    await user.click(trigger);

    const drawer = screen.getByRole("dialog", { name: /Player C vs Player D/ });
    expect(drawer).toBeInTheDocument();
    const closeButton = within(drawer).getByRole("button", { name: "Close match details" });
    const marketLink = within(drawer).getByRole("link", { name: "View market ↗" });
    expect(closeButton).toHaveFocus();
    expect(document.body).toHaveStyle({ overflow: "hidden" });
    await user.tab({ shift: true });
    expect(marketLink).toHaveFocus();
    fireEvent.keyDown(marketLink, { key: "Tab" });
    expect(closeButton).toHaveFocus();
    expect(within(drawer).getByText("Rank #12")).toBeInTheDocument();
    expect(within(drawer).getByText("Rank #18")).toBeInTheDocument();
    expect(within(drawer).queryByLabelText("Player ranks")).not.toBeInTheDocument();
    expect(within(drawer).queryByText("Skill gap")).not.toBeInTheDocument();
    expect(within(drawer).queryByText("Match skill")).not.toBeInTheDocument();
    expect(within(drawer).getByRole("heading", { name: "Prediction" })).toBeInTheDocument();
    const outcomeSection = within(drawer).getByRole("region", { name: "Outcome" });
    expect(within(outcomeSection).getByText("Prediction")).toBeInTheDocument();
    expect(within(outcomeSection).queryByText("Result")).not.toBeInTheDocument();
    expect(within(drawer).getByText("Model vs Polymarket")).toBeInTheDocument();
    expect(within(drawer).getByRole("columnheader", { name: "Outcome" })).toBeInTheDocument();
    expect(within(drawer).getByRole("columnheader", { name: "Difference" })).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: /Player C vs Player D/ })).not.toBeInTheDocument();
    expect(document.body).not.toHaveStyle({ overflow: "hidden" });
    expect(trigger).toHaveFocus();
  });

  it("renders all completed forecasts and rankings", async () => {
    const user = userEvent.setup();
    const largePredictions: PredictionPayload = {
      ...predictions,
      matches: Array.from({ length: 105 }, (_, index) => ({
        ...predictions.matches[0],
        id: `archive-${index}`,
        timestamp: 1100 + index,
        tournament: `Archive ${index}`,
      })),
      top_players: Array.from({ length: 105 }, (_, index) => ({
        rank: index + 1,
        name: `Ranked Player ${index + 1}`,
        skill: 2 - index / 50,
        variance: 0.2,
      })),
    };
    vi.stubGlobal("fetch", vi.fn((url: string) => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(url.includes("results") ? results : largePredictions),
    })));

    render(<App />);
    await waitFor(() => expect(screen.getByText("Upcoming predictions")).toBeInTheDocument());

    const completedSection = screen.getByRole("region", { name: "Completed results" });
    const rankingsSection = screen.getByRole("region", { name: "Player rankings" });
    expect(within(completedSection).getAllByRole("article")).toHaveLength(105);
    expect(within(rankingsSection).getAllByRole("article")).toHaveLength(105);
    expect(within(completedSection).getByText("Showing 105 of 105 matches")).toBeInTheDocument();
    expect(within(rankingsSection).getByText("Showing 105 of 105 players")).toBeInTheDocument();
    expect(within(completedSection).getByText("Archive 104")).toBeInTheDocument();
    expect(within(completedSection).getByText("Archive 0")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show 100 more" })).not.toBeInTheDocument();

    const search = screen.getByLabelText("Search");
    await user.type(search, "Archive 104");
    await waitFor(() => expect(within(completedSection).getAllByRole("article")).toHaveLength(1));
    await user.clear(search);
    await waitFor(() => expect(within(completedSection).getAllByRole("article")).toHaveLength(105));
  });
});
