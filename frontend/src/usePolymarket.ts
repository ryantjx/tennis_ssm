import { useEffect, useMemo, useState } from "react";
import type { MarketPrediction, MatchPrediction } from "./types";
import { matchId } from "./utils";

export type PolymarketStatus = "loading" | "current" | "fallback";

interface PolymarketState {
  predictions: Record<string, MarketPrediction>;
  status: PolymarketStatus;
  lastCheckedAt: string | null;
}

interface PolymarketSource {
  dataUrl: string;
  tagSlug: string;
}

interface TennisMarketCandidate extends MarketPrediction {
  event_date?: string;
}

const POLYMARKET_SITE_URL = "https://polymarket.com";
let requestSequence = 0;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function canonicalPlayer(name: unknown): string {
  if (!name) return "";
  const normalized = String(name).normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  const cleaned = normalized
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
  return cleaned.split(/\s+/).filter(Boolean).join(" ");
}

function playerPairKey(player1: unknown, player2: unknown): string | null {
  const p1 = canonicalPlayer(player1);
  const p2 = canonicalPlayer(player2);
  if (!p1 || !p2 || p1 === p2) return null;
  return [p1, p2].sort().join("|");
}

function jsonStringList(value: unknown): string[] | null {
  if (Array.isArray(value) && value.every((item) => typeof item === "string")) return value;
  if (typeof value !== "string") return null;
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) && parsed.every((item) => typeof item === "string")
      ? parsed
      : null;
  } catch {
    return null;
  }
}

function numberOrNull(value: unknown): number | null {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

function eventSlugUrl(slug: unknown): string | undefined {
  return typeof slug === "string" && slug
    ? `${POLYMARKET_SITE_URL}/event/${slug}`
    : undefined;
}

function marketUpdatedAt(market: Record<string, unknown>, event: Record<string, unknown>): string | undefined {
  const updatedAt = market.updatedAt ?? event.updatedAt;
  return typeof updatedAt === "string" ? updatedAt : undefined;
}

function marketEventDate(event: Record<string, unknown>): string | undefined {
  return typeof event.eventDate === "string" ? event.eventDate.slice(0, 10) : undefined;
}

function marketSelection(market: Record<string, unknown>): string | null {
  if (isRecord(market.marketMetadata) && typeof market.marketMetadata.opticOddsSelection === "string") {
    return market.marketMetadata.opticOddsSelection;
  }
  return typeof market.groupItemTitle === "string" ? market.groupItemTitle : null;
}

function containsDoublesSignal(...values: unknown[]): boolean {
  return values.some((value) => {
    if (typeof value !== "string") return false;
    return /\/|\bdoubles\b/i.test(value);
  });
}

function yesPrice(market: Record<string, unknown>): number | null {
  const outcomes = jsonStringList(market.outcomes);
  const prices = jsonStringList(market.outcomePrices);
  if (!outcomes || !prices || outcomes.length !== prices.length || outcomes.filter((item) => item === "Yes").length !== 1) {
    return null;
  }
  const price = numberOrNull(prices[outcomes.indexOf("Yes")]);
  return price !== null && price >= 0 && price <= 1 ? price : null;
}

function directMoneylineCandidate(
  market: Record<string, unknown>,
  event: Record<string, unknown>,
): TennisMarketCandidate | null {
  const outcomes = jsonStringList(market.outcomes);
  const prices = jsonStringList(market.outcomePrices);
  if (!outcomes || !prices || outcomes.length !== 2 || prices.length !== 2) return null;
  if (outcomes.includes("Yes") || outcomes.includes("No")) return null;
  const [outcome1, outcome2] = outcomes;
  if (containsDoublesSignal(event.title, market.question, outcome1, outcome2)) return null;
  const price1 = numberOrNull(prices[0]);
  const price2 = numberOrNull(prices[1]);
  if (price1 === null || price2 === null || price1 < 0 || price1 > 1 || price2 < 0 || price2 > 1) return null;

  return {
    source: "polymarket",
    event_id: String(event.id ?? ""),
    event_title: typeof event.title === "string" ? event.title : undefined,
    event_slug: typeof event.slug === "string" ? event.slug : undefined,
    event_url: eventSlugUrl(event.slug),
    market_id: String(market.id ?? ""),
    market_slug: typeof market.slug === "string" ? market.slug : undefined,
    market_question: typeof market.question === "string" ? market.question : undefined,
    outcome1,
    outcome2,
    price1,
    price2,
    player1_price: price1,
    player2_price: price2,
    updated_at: marketUpdatedAt(market, event),
    volume: numberOrNull(market.volume ?? event.volume),
    liquidity: numberOrNull(market.liquidity ?? event.liquidity),
    event_date: marketEventDate(event),
  };
}

interface GroupedYesNoMarket {
  event: Record<string, unknown>;
  markets: Record<string, { market: Record<string, unknown>; price: number }>;
}

function eventGroupKey(event: Record<string, unknown>): string | null {
  if (typeof event.slug === "string" && event.slug) return event.slug;
  if (typeof event.id === "string" && event.id) return event.id;
  if (typeof event.title === "string" && event.title) return event.title;
  return null;
}

function yesNoMoneylineCandidate(group: GroupedYesNoMarket): TennisMarketCandidate | null {
  const entries = Object.entries(group.markets);
  if (entries.length !== 2) return null;
  const [[outcome1, first], [outcome2, second]] = entries;
  if (containsDoublesSignal(group.event.title, outcome1, outcome2)) return null;
  const updatedAt = [marketUpdatedAt(first.market, group.event), marketUpdatedAt(second.market, group.event)]
    .filter((value): value is string => typeof value === "string")
    .sort()
    .at(-1);

  return {
    source: "polymarket",
    event_id: String(group.event.id ?? ""),
    event_title: typeof group.event.title === "string" ? group.event.title : undefined,
    event_slug: typeof group.event.slug === "string" ? group.event.slug : undefined,
    event_url: eventSlugUrl(group.event.slug),
    market_id: String(first.market.id ?? ""),
    market_slug: typeof first.market.slug === "string" ? first.market.slug : undefined,
    market_question: typeof group.event.title === "string" ? group.event.title : undefined,
    outcome1,
    outcome2,
    price1: first.price,
    price2: second.price,
    player1_price: first.price,
    player2_price: second.price,
    updated_at: updatedAt,
    volume: numberOrNull(first.market.volume ?? second.market.volume ?? group.event.volume),
    liquidity: numberOrNull(first.market.liquidity ?? second.market.liquidity ?? group.event.liquidity),
    event_date: marketEventDate(group.event),
  };
}

export function parsePolymarketTennisMarkets(events: unknown[]): Record<string, TennisMarketCandidate[]> {
  const candidates: Record<string, TennisMarketCandidate[]> = {};
  const yesNoGroups: Record<string, GroupedYesNoMarket> = {};

  for (const event of events) {
    if (!isRecord(event) || !Array.isArray(event.markets)) continue;
    for (const marketValue of event.markets) {
      if (!isRecord(marketValue)) continue;
      if (marketValue.sportsMarketType !== "moneyline" || marketValue.active !== true || marketValue.closed === true) continue;

      const direct = directMoneylineCandidate(marketValue, event);
      if (direct) {
        const key = playerPairKey(direct.outcome1, direct.outcome2);
        if (key) {
          candidates[key] ??= [];
          candidates[key].push(direct);
        }
        continue;
      }

      const selection = marketSelection(marketValue);
      const price = yesPrice(marketValue);
      const key = eventGroupKey(event);
      if (!selection || selection.toLowerCase() === "draw" || price === null || !key) continue;
      yesNoGroups[key] ??= { event, markets: {} };
      if (selection in yesNoGroups[key].markets) continue;
      yesNoGroups[key].markets[selection] = { market: marketValue, price };
    }
  }

  for (const group of Object.values(yesNoGroups)) {
    const candidate = yesNoMoneylineCandidate(group);
    const key = candidate ? playerPairKey(candidate.outcome1, candidate.outcome2) : null;
    if (!candidate || !key) continue;
    candidates[key] ??= [];
    candidates[key].push(candidate);
  }

  return candidates;
}

function dateDeltaDays(left: string, right: string): number | null {
  const leftTime = Date.parse(`${left.slice(0, 10)}T12:00:00Z`);
  const rightTime = Date.parse(`${right.slice(0, 10)}T12:00:00Z`);
  if (!Number.isFinite(leftTime) || !Number.isFinite(rightTime)) return null;
  return Math.abs(leftTime - rightTime) / 86_400_000;
}

function tokenOverlap(left: string | undefined, right: string | undefined): number {
  if (!left || !right) return 0;
  const leftTokens = new Set(canonicalPlayer(left).split(" ").filter((token) => token.length > 2));
  return canonicalPlayer(right)
    .split(" ")
    .filter((token) => token.length > 2 && leftTokens.has(token))
    .length;
}

function candidateScore(match: MatchPrediction, candidate: TennisMarketCandidate): number {
  let score = 0;
  if (candidate.event_date) {
    const delta = dateDeltaDays(match.date, candidate.event_date);
    if (delta !== null && delta > 2) return Number.NEGATIVE_INFINITY;
    if (delta !== null) score += 100 - delta * 25;
  }
  score += tokenOverlap(match.tournament, candidate.event_title) * 5;
  score += (candidate.liquidity ?? 0) / 1_000_000;
  score += (candidate.volume ?? 0) / 10_000_000;
  if (candidate.updated_at) score += Date.parse(candidate.updated_at) / 1e16;
  return score;
}

function bestCandidate(match: MatchPrediction, candidates: TennisMarketCandidate[]): TennisMarketCandidate | null {
  const ranked = candidates
    .map((candidate) => ({ candidate, score: candidateScore(match, candidate) }))
    .filter(({ score }) => Number.isFinite(score))
    .sort((left, right) => right.score - left.score);
  return ranked[0]?.candidate ?? null;
}

export function predictionsForMatches(matches: MatchPrediction[], events: unknown[]): Record<string, MarketPrediction> {
  const markets = parsePolymarketTennisMarkets(events);
  return Object.fromEntries(matches.flatMap((match) => {
    if (match.actual_winner) return [];
    const key = playerPairKey(match.player1, match.player2);
    if (!key || !markets[key]) return [];
    const market = bestCandidate(match, markets[key]);
    if (!market) return [];
    const player1Key = canonicalPlayer(match.player1);
    const marketOutcome1Key = canonicalPlayer(market.outcome1);
    const player1Price = player1Key === marketOutcome1Key ? market.price1 : market.price2;
    const player2Price = player1Key === marketOutcome1Key ? market.price2 : market.price1;
    return [[matchId(match), {
      ...market,
      player1_market_name: player1Key === marketOutcome1Key ? market.outcome1 : market.outcome2,
      player2_market_name: player1Key === marketOutcome1Key ? market.outcome2 : market.outcome1,
      player1_price: player1Price ?? 0,
      player2_price: player2Price ?? 0,
      player1_edge: Math.round((match.p_player1_win - (player1Price ?? 0)) * 10_000) / 10_000,
      player2_edge: Math.round((match.p_player2_win - (player2Price ?? 0)) * 10_000) / 10_000,
      matched_by: "live_canonical_player_pair",
    } satisfies MarketPrediction]];
  }));
}

function fallbackPredictions(matches: MatchPrediction[]): Record<string, MarketPrediction> {
  return Object.fromEntries(matches.flatMap((match) =>
    match.market && !match.actual_winner ? [[matchId(match), match.market] as const] : [],
  ));
}

async function fetchAllEvents(source: PolymarketSource): Promise<unknown[]> {
  const events: unknown[] = [];
  let cursor: string | undefined;

  do {
    const url = new URL(source.dataUrl);
    url.searchParams.set("limit", "100");
    url.searchParams.set("tag_slug", source.tagSlug);
    url.searchParams.set("closed", "false");
    url.searchParams.set("decimalized", "true");
    url.searchParams.set("refresh", `${Date.now()}-${requestSequence++}`);
    if (cursor) url.searchParams.set("after_cursor", cursor);

    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`Polymarket request failed with ${response.status}`);
    const page: unknown = await response.json();
    if (!isRecord(page) || !Array.isArray(page.events)) throw new Error("Polymarket response is invalid");
    events.push(...page.events);
    if (page.next_cursor !== undefined && typeof page.next_cursor !== "string") throw new Error("Polymarket cursor is invalid");
    cursor = page.next_cursor;
  } while (cursor);

  return events;
}

export function usePolymarket(matches: MatchPrediction[], source: PolymarketSource | undefined): PolymarketState {
  const fallback = useMemo(() => fallbackPredictions(matches), [matches]);
  const [state, setState] = useState<PolymarketState>({
    predictions: fallback,
    status: source ? "loading" : "fallback",
    lastCheckedAt: null,
  });

  useEffect(() => {
    let active = true;
    if (!source || matches.length === 0) {
      setState({ predictions: fallback, status: "fallback", lastCheckedAt: null });
      return () => { active = false; };
    }

    setState({ predictions: fallback, status: "loading", lastCheckedAt: null });
    fetchAllEvents(source)
      .then((events) => {
        if (!active) return;
        setState({
          predictions: predictionsForMatches(matches, events),
          status: "current",
          lastCheckedAt: new Date().toISOString(),
        });
      })
      .catch(() => {
        if (!active) return;
        setState({ predictions: fallback, status: "fallback", lastCheckedAt: new Date().toISOString() });
      });

    return () => { active = false; };
  }, [fallback, matches, source?.dataUrl, source?.tagSlug]);

  return state;
}
