export interface MarketPrediction {
  source: "polymarket" | string;
  event_id?: string;
  event_title?: string;
  event_slug?: string;
  event_url?: string;
  market_id?: string;
  market_slug?: string;
  market_question?: string;
  outcome1?: string;
  outcome2?: string;
  price1?: number;
  price2?: number;
  player1_market_name?: string;
  player2_market_name?: string;
  player1_price: number;
  player2_price: number;
  player1_edge?: number;
  player2_edge?: number;
  updated_at?: string;
  volume?: number | null;
  liquidity?: number | null;
  matched_by?: string;
}

export interface MatchPrediction {
  id?: string;
  date: string;
  timestamp: number;
  player1: string;
  player2: string;
  p_player1_win: number;
  p_player2_win: number;
  predicted_winner: string;
  actual_winner: string | null;
  correct: boolean | null;
  confidence: number;
  log_score: number | null;
  player1_skill: number;
  player2_skill: number;
  player1_skill_sd: number;
  player2_skill_sd: number;
  player1_rank?: number | null;
  player2_rank?: number | null;
  player1_latest_skill?: number | null;
  player2_latest_skill?: number | null;
  player1_latest_variance?: number | null;
  player2_latest_variance?: number | null;
  market?: MarketPrediction | null;
  tournament: string;
  location: string;
  tier: string;
  surface: string;
  round: string;
  source?: string;
  source_match_id?: string;
  match_state?: string;
  match_status?: "upcoming" | "in_progress" | "suspended" | string;
  is_future?: boolean;
}

export interface PlayerRanking {
  rank?: number;
  name: string;
  skill: number;
  variance: number;
}

export interface PredictionPayload {
  generated_at: string;
  data_windows?: Record<string, string>;
  model_params: {
    tau: number;
    s: number;
    init_var: number;
  };
  seed_model_params?: {
    tau: number;
    s: number;
    init_var: number;
  };
  optimization?: {
    objective: string;
    selection_note?: string;
    candidate_count?: number;
    best_params?: PredictionPayload["model_params"];
    best_metrics?: Record<string, number>;
  };
  fixture_status?: {
    source?: string;
    start_date?: string;
    end_date?: string;
    loaded?: number;
    matched_model_players?: number;
    skipped_unknown_players?: number;
    synthetic_fallback?: boolean;
    error?: string;
  };
  market_status?: {
    source?: string;
    source_url?: string;
    api_url?: string;
    loaded_events?: number;
    loaded_moneylines?: number;
    matched_model_matches?: number;
    error?: string;
  };
  metrics: {
    n_test_matches: number;
    n_future_matches?: number;
    accuracy: number;
    avg_log_score: number;
    uniform_baseline: number;
  };
  top_players: PlayerRanking[];
  matches: MatchPrediction[];
  future_matches: MatchPrediction[];
}

export interface CompletedResult {
  id: string;
  date: string;
  winner: string | null;
  loser: string | null;
  player1: string;
  player2: string;
  actual_winner: string | null;
  predicted_winner?: string;
  p_player1_win?: number;
  p_player2_win?: number;
  confidence?: number;
  player1_rank?: number | null;
  player2_rank?: number | null;
  player1_latest_skill?: number | null;
  player2_latest_skill?: number | null;
  player1_latest_variance?: number | null;
  player2_latest_variance?: number | null;
  market?: MarketPrediction | null;
  match_status?: string;
  match_state?: string;
  tournament: string;
  location: string;
  tier: string;
  surface: string;
  round: string;
  source?: string;
  source_match_id?: string;
}

export interface ResultsPayload {
  generated_at: string;
  source: string;
  data_window?: {
    start: string;
    end: string;
  };
  results: CompletedResult[];
  current_matches?: CompletedResult[];
}

export interface MatchFilters {
  query: string;
}
