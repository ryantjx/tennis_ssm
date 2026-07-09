import type { MatchPrediction } from "../types";
import {
  confidenceLabel,
  formatDate,
  formatPercent,
  formatSigned,
  probabilityForPredictedWinner,
} from "../utils";
import { MarketComparisonTable } from "./MarketComparisonTable";

interface MatchDetailDrawerProps {
  match: MatchPrediction | null;
  onClose: () => void;
}

export function MatchDetailDrawer({ match, onClose }: MatchDetailDrawerProps) {
  if (!match) return null;

  const predictedProbability = probabilityForPredictedWinner(match);
  const skillGap = match.player1_skill - match.player2_skill;

  return (
    <div className="drawer-backdrop" role="presentation" onClick={onClose}>
      <aside
        className="match-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="drawer-title"
        onClick={(event) => event.stopPropagation()}
      >
        <button className="drawer-close" type="button" onClick={onClose} aria-label="Close match details">Close</button>
        <span className="eyebrow">{match.tournament} · {match.surface}</span>
        <h2 id="drawer-title">{match.player1} vs {match.player2}</h2>
        <p>{formatDate(match.date)} · {match.location} · {match.round}</p>

        <div className="drawer-pick">
          <span>Model pick</span>
          <strong>{match.predicted_winner}</strong>
          <b>{confidenceLabel(match.confidence)} {formatPercent(predictedProbability, 1)}</b>
        </div>

        <div className="probability-detail">
          <ProbabilityRow label={match.player1} value={match.p_player1_win} />
          <ProbabilityRow label={match.player2} value={match.p_player2_win} />
        </div>

        <div className="drawer-player-stats" aria-label="Player ranks and skills">
          <PlayerStatCard
            name={match.player1}
            rank={match.player1_rank}
            latestSkill={match.player1_latest_skill}
            latestVariance={match.player1_latest_variance}
            matchSkill={match.player1_skill}
            matchSkillSd={match.player1_skill_sd}
          />
          <PlayerStatCard
            name={match.player2}
            rank={match.player2_rank}
            latestSkill={match.player2_latest_skill}
            latestVariance={match.player2_latest_variance}
            matchSkill={match.player2_skill}
            matchSkillSd={match.player2_skill_sd}
          />
        </div>

        {match.market ? (
          <section className="drawer-section" aria-label="Market comparison">
            <MarketComparisonTable match={match} />
          </section>
        ) : null}

        <dl className="detail-grid">
          <div>
            <dt>Actual winner</dt>
            <dd>{match.actual_winner ?? "Not yet played"}</dd>
          </div>
          <div>
            <dt>Outcome</dt>
            <dd>{match.is_future ? "Upcoming" : match.correct ? "Correct" : "Wrong"}</dd>
          </div>
          <div>
            <dt>Log score</dt>
            <dd>{match.log_score === null ? "Pending" : match.log_score.toFixed(4)}</dd>
          </div>
          <div>
            <dt>Skill gap</dt>
            <dd>{formatSigned(skillGap, 2)} for {skillGap >= 0 ? match.player1 : match.player2}</dd>
          </div>
          <div>
            <dt>{match.player1} skill</dt>
            <dd>{formatSigned(match.player1_skill, 2)} sd {match.player1_skill_sd.toFixed(2)}</dd>
          </div>
          <div>
            <dt>{match.player2} skill</dt>
            <dd>{formatSigned(match.player2_skill, 2)} sd {match.player2_skill_sd.toFixed(2)}</dd>
          </div>
        </dl>
      </aside>
    </div>
  );
}

function PlayerStatCard({
  name,
  rank,
  latestSkill,
  latestVariance,
  matchSkill,
  matchSkillSd,
}: {
  name: string;
  rank?: number | null;
  latestSkill?: number | null;
  latestVariance?: number | null;
  matchSkill: number;
  matchSkillSd: number;
}) {
  const latestSd = typeof latestVariance === "number" ? Math.sqrt(latestVariance) : null;
  const combinedSkill = typeof latestSkill === "number" ? latestSkill : matchSkill;
  return (
    <div className="drawer-player-stat">
      <span className="eyebrow">{name}</span>
      <strong>{rank ? `Rank #${rank}` : "Rank unavailable"}</strong>
      <dl>
        <div>
          <dt>Skill</dt>
          <dd>
            {formatSigned(combinedSkill, 2)}
            {latestSd !== null ? ` ±${latestSd.toFixed(2)}` : ""}
          </dd>
        </div>
        <div>
          <dt>Match skill</dt>
          <dd>{formatSigned(matchSkill, 2)} ±{matchSkillSd.toFixed(2)}</dd>
        </div>
      </dl>
    </div>
  );
}

function ProbabilityRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="probability-row">
      <div>
        <span>{label}</span>
        <strong>{formatPercent(value, 1)}</strong>
      </div>
      <span className="probability-row__track">
        <span style={{ width: `${value * 100}%` }} />
      </span>
    </div>
  );
}
