import type { MouseEvent } from "react";
import type { MatchPrediction } from "../types";
import { formatPercent, formatSigned } from "../utils";

interface MarketComparisonTableProps {
  match: MatchPrediction;
  compact?: boolean;
}

export function MarketComparisonTable({ match, compact = false }: MarketComparisonTableProps) {
  if (!match.market) return null;

  const rows = [
    {
      outcome: `${match.player1} win`,
      model: match.p_player1_win,
      market: match.market.player1_price,
    },
    {
      outcome: `${match.player2} win`,
      model: match.p_player2_win,
      market: match.market.player2_price,
    },
  ];

  function stopCardOpen(event: MouseEvent) {
    event.stopPropagation();
  }

  return (
    <section
      className={`market-comparison ${compact ? "market-comparison--compact" : ""}`}
      aria-label="Model vs Polymarket"
      onClick={stopCardOpen}
      onKeyDown={(event) => event.stopPropagation()}
    >
      <div className="market-comparison__title">
        <h3>Model vs Polymarket</h3>
        {match.market.event_url ? (
          <a href={match.market.event_url} target="_blank" rel="noreferrer">
            View market ↗
          </a>
        ) : null}
      </div>
      <div className="market-table-wrap">
        <table className="market-table">
          <thead>
            <tr>
              <th>Outcome</th>
              <th>Model</th>
              <th>Market</th>
              <th>Difference</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.outcome}>
                <td><strong>{row.outcome}</strong></td>
                <td>{formatPercent(row.model, 1)}</td>
                <td>{formatPercent(row.market, 1)}</td>
                <td>{formatSigned((row.market - row.model) * 100, 1)} pp</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!compact ? (
        <p className="drawer-note">Market-implied probabilities are live prices and may not total exactly 100%.</p>
      ) : null}
    </section>
  );
}
