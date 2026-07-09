import type { PlayerRanking } from "../types";
import { formatSigned } from "../utils";

interface PlayerRankingsProps {
  players: PlayerRanking[];
}

export function PlayerRankings({ players }: PlayerRankingsProps) {
  return (
    <section className="section" id="rankings" aria-labelledby="rankings-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Filtered skill state</span>
          <h2 id="rankings-title">Player rankings</h2>
        </div>
        <p>Posterior skill estimates after the historical training window.</p>
      </div>
      <div className="ranking-chart">
        {players.map((player, index) => (
          <article className="ranking-row" key={player.name}>
            <span className="ranking-row__rank">{index + 1}</span>
            <strong>{player.name}</strong>
            <SkillScale player={player} />
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

function SkillScale({ player }: { player: PlayerRanking }) {
  const maxSkill = 5;
  const sd = Math.sqrt(player.variance);
  const skill = clamp(player.skill, 0, maxSkill);
  const intervalStart = clamp(player.skill - sd, 0, maxSkill);
  const intervalEnd = clamp(player.skill + sd, 0, maxSkill);
  const point = (skill / maxSkill) * 100;
  const start = (intervalStart / maxSkill) * 100;
  const width = Math.max(1, ((intervalEnd - intervalStart) / maxSkill) * 100);

  return (
    <div
      className="skill-scale"
      aria-label={`${player.name} skill ${player.skill.toFixed(2)}, one standard deviation ${sd.toFixed(2)}`}
    >
      <div className="skill-scale__axis" aria-hidden="true">
        {[1, 2, 3, 4, 5].map((tick) => (
          <span key={tick} style={{ left: `${(tick / maxSkill) * 100}%` }}>
            {tick}
          </span>
        ))}
      </div>
      <div className="skill-scale__track">
        <span className="skill-scale__interval" style={{ left: `${start}%`, width: `${width}%` }} />
        <span className="skill-scale__bar" style={{ width: `${point}%` }} />
        <span className="skill-scale__point" style={{ left: `${point}%` }} />
      </div>
    </div>
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
