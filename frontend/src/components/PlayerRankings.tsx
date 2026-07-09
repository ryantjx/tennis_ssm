import { useMemo } from "react";
import type { PlayerRanking } from "../types";
import { formatSigned } from "../utils";

interface PlayerRankingsProps {
  players: PlayerRanking[];
}

export function PlayerRankings({ players }: PlayerRankingsProps) {
  const domain = useMemo(() => skillDomain(players), [players]);

  return (
    <section className="section" id="rankings" aria-labelledby="rankings-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">All trained players</span>
          <h2 id="rankings-title">Player rankings</h2>
        </div>
        <p>Posterior skill estimates after the historical training window.</p>
      </div>
      <div className="ranking-tools">
        <p>{players.length} players shown</p>
      </div>
      <div className="ranking-chart">
        {players.map((player, index) => (
          <article className={`ranking-row ${player.skill < 0 ? "ranking-row--negative" : ""}`} key={player.name}>
            <span className="ranking-row__rank">#{player.rank ?? index + 1}</span>
            <strong className="ranking-row__player">
              <span>{player.name}</span>
            </strong>
            <SkillScale player={player} domain={domain} />
            <b>
              {formatSigned(player.skill, 2)}
              <span>±{Math.sqrt(player.variance).toFixed(2)}</span>
            </b>
          </article>
        ))}
      </div>
    </section>
  );
}

function SkillScale({ player, domain }: { player: PlayerRanking; domain: number }) {
  const sd = Math.sqrt(player.variance);
  const intervalStart = clamp(player.skill - sd, -domain, domain);
  const intervalEnd = clamp(player.skill + sd, -domain, domain);
  const point = toPercent(clamp(player.skill, -domain, domain), domain);
  const zero = toPercent(0, domain);
  const start = toPercent(intervalStart, domain);
  const end = toPercent(intervalEnd, domain);
  const fillLeft = Math.min(point, zero);
  const fillWidth = Math.max(1, Math.abs(point - zero));
  const ticks = [-domain, -domain / 2, 0, domain / 2, domain];

  return (
    <div
      className="skill-scale"
      aria-label={`${player.name} skill ${player.skill.toFixed(2)}, one standard deviation ${sd.toFixed(2)}`}
    >
      <div className="skill-scale__axis" aria-hidden="true">
        {ticks.map((tick) => (
          <span key={tick} style={{ left: `${toPercent(tick, domain)}%` }}>
            {tick === 0 ? "0" : formatSigned(tick, 1)}
          </span>
        ))}
      </div>
      <div className="skill-scale__track">
        <span className="skill-scale__zero" style={{ left: `${zero}%` }} />
        <span className="skill-scale__interval" style={{ left: `${start}%`, width: `${Math.max(1, end - start)}%` }} />
        <span className="skill-scale__bar" style={{ left: `${fillLeft}%`, width: `${fillWidth}%` }} />
        <span className="skill-scale__point" style={{ left: `${point}%` }} />
      </div>
    </div>
  );
}

function skillDomain(players: PlayerRanking[]): number {
  const maxAbs = players.reduce((max, player) => {
    const sd = Math.sqrt(player.variance);
    return Math.max(max, Math.abs(player.skill - sd), Math.abs(player.skill + sd));
  }, 1);
  return Math.ceil(maxAbs * 2) / 2;
}

function toPercent(value: number, domain: number): number {
  return ((value + domain) / (domain * 2)) * 100;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
