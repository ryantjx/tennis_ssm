import { useMemo, useState } from "react";
import {
  buildLinePath,
  collapseByDate,
  createScale,
  dateTicks,
  dateToDay,
  dayToDate,
  DEFAULT_MAX_GAP_DAYS,
  distributeLabels,
  filterByWindow,
  findGaps,
  formatAxisDate,
  LABEL_MIN_GAP,
  nearestSamples,
  skillDomain,
  skillTicks,
  type Scale,
} from "../chartUtils";
import type { SkillHistory, SkillHistoryPlayer } from "../types";
import { formatSigned } from "../utils";

const WIDTH = 900;
const HEIGHT = 380;
const MARGIN = { top: 30, right: 112, bottom: 38, left: 46 };
const PLOT_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;

/**
 * How stale a player's final observation can be before it is flagged.
 *
 * The WTA calendar has an off-season and injury lay-offs, so a lower threshold
 * would mark much of the top ten during the winter break. Thirty days is long
 * enough to clear a normal tournament gap and short enough to catch a genuine
 * absence within the chart window.
 */
const INACTIVE_DAYS = 30;

/**
 * A fixed categorical palette. Ten distinct hues are required, so these are
 * ordered for maximum separation rather than for semantic meaning.
 */
const SERIES_COLOURS = [
  "#17633b",
  "#2f6f9f",
  "#bc6246",
  "#7b5ea7",
  "#b08b1f",
  "#1f8a70",
  "#c2417a",
  "#4b6b8f",
  "#8a6d3b",
  "#5d8a48",
];

interface SkillHistoryChartProps {
  history?: SkillHistory;
}

export function SkillHistoryChart({ history }: SkillHistoryChartProps) {
  const [activeRank, setActiveRank] = useState<number | null>(null);
  const [hoverDay, setHoverDay] = useState<number | null>(null);

  const model = useMemo(() => buildChartModel(history), [history]);

  if (!model) {
    return (
      <div className="empty-state">
        <strong>No skill history is available yet.</strong>
        <span>
          History is exported with each model run. The next run will populate this chart.
        </span>
      </div>
    );
  }

  const { players, slams, windowStart, windowEnd, x, y, ticksX, ticksY } = model;
  const showZeroLine = y.domain[0] <= 0 && y.domain[1] >= 0;

  const samples = hoverDay === null ? [] : nearestSamples(players, hoverDay);
  const crosshairX = hoverDay === null ? null : x(hoverDay);
  const highlighted = activeRank === null
    ? null
    : players.find((player) => player.rank === activeRank) ?? null;

  return (
    <figure className="skill-chart">
      <svg
        className="skill-chart__svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-labelledby="skill-chart-title skill-chart-desc"
        onMouseLeave={() => setHoverDay(null)}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          // jsdom and a zero-width layout both report a zero-size box; without
          // this guard the scale inversion yields NaN and the readout breaks.
          if (!rect.width) return;
          const offset = ((event.clientX - rect.left) / rect.width) * WIDTH;
          const day = Math.round(x.invert(offset));
          if (Number.isFinite(day)) setHoverDay(clamp(day, windowStart, windowEnd));
        }}
      >
        <title id="skill-chart-title">Historical skill trajectories for the top ten players</title>
        <desc id="skill-chart-desc">
          Latent skill estimates from the model for the ten highest-ranked players over the last
          twelve months, with shaded bands marking the Grand Slam tournaments.
        </desc>

        <defs>
          <clipPath id="skill-chart-clip">
            <rect x={MARGIN.left} y={MARGIN.top} width={PLOT_WIDTH} height={PLOT_HEIGHT} />
          </clipPath>
        </defs>

        {slams.map((slam) => {
          const left = x(dateToDay(slam.start));
          const right = x(dateToDay(slam.end));
          const width = Math.max(right - left, 2);
          return (
            <g key={slam.key} className="skill-chart__slam">
              <rect
                x={left}
                y={MARGIN.top}
                width={width}
                height={PLOT_HEIGHT}
                aria-label={`${slam.name}, ${slam.start} to ${slam.end}`}
              >
                <title>{`${slam.name}: ${slam.start} to ${slam.end}`}</title>
              </rect>
              <text
                x={left + width / 2}
                y={MARGIN.top - 10}
                textAnchor="middle"
                className="skill-chart__slam-label"
              >
                {slam.short}
              </text>
            </g>
          );
        })}

        {ticksY.map((tick) => (
          <g key={`y-${tick}`}>
            <line
              className={
                tick === 0 && showZeroLine
                  ? "skill-chart__grid skill-chart__grid--zero"
                  : "skill-chart__grid"
              }
              x1={MARGIN.left}
              x2={MARGIN.left + PLOT_WIDTH}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text
              className="skill-chart__axis-label"
              x={MARGIN.left - 8}
              y={y(tick)}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {tick === 0 ? "0" : formatSigned(tick, 1)}
            </text>
          </g>
        ))}

        <text
          className="skill-chart__axis-title"
          transform={`rotate(-90 12 ${MARGIN.top + PLOT_HEIGHT / 2})`}
          x={12}
          y={MARGIN.top + PLOT_HEIGHT / 2}
          textAnchor="middle"
        >
          Model skill, before each match
        </text>

        {ticksX.map((tick) => (
          <g key={`x-${tick}`}>
            <line
              className="skill-chart__grid skill-chart__grid--vertical"
              x1={x(tick)}
              x2={x(tick)}
              y1={MARGIN.top}
              y2={MARGIN.top + PLOT_HEIGHT}
            />
            <text
              className="skill-chart__axis-label"
              x={x(tick)}
              y={MARGIN.top + PLOT_HEIGHT + 18}
              textAnchor="middle"
            >
              {formatAxisDate(tick)}
            </text>
          </g>
        ))}

        <g clipPath="url(#skill-chart-clip)">
          {/* Dashed connectors sit under the lines so a hiatus reads as a break
              in observation rather than a modelled trend. */}
          {players.map((player, index) =>
            findGaps(player.points).map((gap) => {
              const dimmed = highlighted !== null && highlighted.rank !== player.rank;
              return (
                <line
                  key={`gap-${player.name}-${gap.to.date}`}
                  className={dimmed ? "skill-chart__gap skill-chart__gap--dim" : "skill-chart__gap"}
                  x1={x(dateToDay(gap.from.date))}
                  y1={y(gap.from.skill)}
                  x2={x(dateToDay(gap.to.date))}
                  y2={y(gap.to.skill)}
                  stroke={SERIES_COLOURS[index % SERIES_COLOURS.length]}
                  aria-hidden="true"
                />
              );
            }),
          )}

          {players.map((player, index) => {
            const dimmed = highlighted !== null && highlighted.rank !== player.rank;
            return (
              <path
                key={player.name}
                className={dimmed ? "skill-chart__line skill-chart__line--dim" : "skill-chart__line"}
                d={buildLinePath(player.points, x, y, DEFAULT_MAX_GAP_DAYS)}
                stroke={SERIES_COLOURS[index % SERIES_COLOURS.length]}
                data-rank={player.rank}
                data-player={player.name}
              />
            );
          })}

          {/* A player whose last observation predates the window end is drawn
              with a hollow final marker, so the line is not read as current. */}
          {players.map((player, index) => {
            const last = player.points[player.points.length - 1];
            const day = dateToDay(last.date);
            if (day >= windowEnd - INACTIVE_DAYS) return null;
            const dimmed = highlighted !== null && highlighted.rank !== player.rank;
            return (
              <circle
                key={`inactive-${player.name}`}
                className={dimmed ? "skill-chart__stale skill-chart__stale--dim" : "skill-chart__stale"}
                cx={x(day)}
                cy={y(last.skill)}
                r={3.5}
                stroke={SERIES_COLOURS[index % SERIES_COLOURS.length]}
              >
                <title>{`${player.name}: no observation since ${last.date}`}</title>
              </circle>
            );
          })}

          {crosshairX !== null && (
            <line
              className="skill-chart__crosshair"
              x1={crosshairX}
              x2={crosshairX}
              y1={MARGIN.top}
              y2={MARGIN.top + PLOT_HEIGHT}
            />
          )}

          {samples.map((sample) => {
            const player = players[sample.playerIndex];
            if (!player) return null;
            const dimmed = highlighted !== null && highlighted.rank !== player.rank;
            return (
              <circle
                key={player.name}
                className={dimmed ? "skill-chart__dot skill-chart__dot--dim" : "skill-chart__dot"}
                cx={x(dateToDay(sample.point.date))}
                cy={y(sample.point.skill)}
                r={dimmed ? 2.5 : 4}
                fill={SERIES_COLOURS[sample.playerIndex % SERIES_COLOURS.length]}
              />
            );
          })}
        </g>

        {model.labels.map((label, index) => (
          <g key={`label-${label.player.name}`} className="skill-chart__end-label">
            <line
              className="skill-chart__leader"
              x1={MARGIN.left + PLOT_WIDTH}
              x2={MARGIN.left + PLOT_WIDTH + 6}
              y1={label.lineY}
              y2={label.labelY}
            />
            <text
              x={MARGIN.left + PLOT_WIDTH + 9}
              y={label.labelY}
              dominantBaseline="middle"
              className={
                highlighted !== null && highlighted.rank !== label.player.rank
                  ? "skill-chart__end-label-text skill-chart__end-label-text--dim"
                  : "skill-chart__end-label-text"
              }
              fill={SERIES_COLOURS[index % SERIES_COLOURS.length]}
            >
              {`#${label.player.rank} ${shortName(label.player.name)}`}
            </text>
          </g>
        ))}
      </svg>

      <figcaption className="skill-chart__caption">
        <span>
          Skill is the model&rsquo;s latent estimate before each match, on a dimensionless scale
          centred at zero, so the series reflects the information available at the time rather
          than a retrospective fit. Long gaps in a player&rsquo;s schedule are drawn dashed rather
          than interpolated, and a hollow marker shows the last observation when it is not recent.
        </span>
        <span className="skill-chart__readout" aria-live="polite">
          {hoverDay === null
            ? `Window ${dayToDate(windowStart)} to ${dayToDate(windowEnd)}.`
            : `Nearest observations to ${dayToDate(hoverDay)}.`}
        </span>
      </figcaption>

      <ul className="skill-chart__legend" aria-label="Players">
        {players.map((player, index) => (
          <li key={player.name}>
            <button
              type="button"
              className={
                highlighted?.rank === player.rank
                  ? "skill-chart__legend-item skill-chart__legend-item--active"
                  : "skill-chart__legend-item"
              }
              aria-pressed={highlighted?.rank === player.rank}
              onClick={() =>
                setActiveRank(highlighted?.rank === player.rank ? null : player.rank)
              }
            >
              <span
                className="skill-chart__swatch"
                style={{ background: SERIES_COLOURS[index % SERIES_COLOURS.length] }}
                aria-hidden="true"
              />
              <span className="skill-chart__legend-name">{player.name}</span>
              <span className="skill-chart__legend-rank">#{player.rank}</span>
            </button>
          </li>
        ))}
      </ul>

      {hoverDay !== null && samples.length ? (
        <table className="skill-chart__tooltip" aria-label="Skill at hovered date">
          <caption>{`Nearest observation to ${dayToDate(hoverDay)}`}</caption>
          <thead>
            <tr>
              <th scope="col">Player</th>
              <th scope="col">Skill</th>
              <th scope="col">Date</th>
            </tr>
          </thead>
          <tbody>
            {[...samples]
              .sort((left, right) => players[left.playerIndex].rank - players[right.playerIndex].rank)
              .map((sample) => (
                <tr key={players[sample.playerIndex].name}>
                  <th scope="row">{players[sample.playerIndex].name}</th>
                  <td>{formatSigned(sample.point.skill, 2)}</td>
                  <td>{sample.point.date}</td>
                </tr>
              ))}
          </tbody>
        </table>
      ) : null}
    </figure>
  );
}

interface ChartModel {
  players: SkillHistoryPlayer[];
  slams: SkillHistory["slams"];
  windowStart: number;
  windowEnd: number;
  x: Scale & { invert(value: number): number };
  y: Scale;
  ticksX: number[];
  ticksY: number[];
  labels: Array<{ player: SkillHistoryPlayer; lineY: number; labelY: number }>;
}

function buildChartModel(history?: SkillHistory): ChartModel | null {
  if (!history?.players?.length) return null;

  const start = history.window?.start;
  const end = history.window?.end;
  if (!start || !end) return null;

  const windowStart = dateToDay(start);
  const windowEnd = dateToDay(end);
  if (!Number.isFinite(windowStart) || !Number.isFinite(windowEnd) || windowEnd <= windowStart) {
    return null;
  }

  const players = history.players
    .map((player) => ({
      ...player,
      points: filterByWindow(collapseByDate(player.points ?? []), windowStart, windowEnd),
    }))
    .filter((player) => player.points.length > 0);

  if (!players.length) return null;

  const xScale = createScale([windowStart, windowEnd], [MARGIN.left, MARGIN.left + PLOT_WIDTH]);
  const yScale = createScale(skillDomain(players), [MARGIN.top + PLOT_HEIGHT, MARGIN.top]);

  const x = Object.assign(xScale, {
    invert(value: number): number {
      const [rangeStart, rangeEnd] = xScale.range;
      const [domainStart, domainEnd] = xScale.domain;
      const ratio = (value - rangeStart) / (rangeEnd - rangeStart);
      return domainStart + ratio * (domainEnd - domainStart);
    },
  });

  // Rank 1 sits at the top of the plot, so labels follow rank order.
  const ordered = [...players].sort((left, right) => left.rank - right.rank);

  // Labels must be stacked in the order their lines actually end, not in rank
  // order. A rank-7 player whose season collapsed can finish below rank 10, and
  // stacking by rank there would cross the leader lines.
  const lineYs = ordered.map((player) => yScale(player.points[player.points.length - 1].skill));
  const visualOrder = lineYs
    .map((lineY, index) => ({ lineY, index }))
    .sort((left, right) => left.lineY - right.lineY)
    .map((entry) => entry.index);
  const visualPositions = distributeLabels(
    visualOrder.map((index) => lineYs[index]),
    LABEL_MIN_GAP,
    MARGIN.top,
    MARGIN.top + PLOT_HEIGHT,
  );
  const labelYs = new Array<number>(ordered.length);
  visualOrder.forEach((playerIndex, slot) => {
    labelYs[playerIndex] = visualPositions[slot];
  });

  return {
    players: ordered,
    slams: history.slams ?? [],
    windowStart,
    windowEnd,
    x,
    y: yScale,
    ticksX: dateTicks(windowStart, windowEnd),
    ticksY: skillTicks(yScale.domain),
    labels: ordered.map((player, index) => ({
      player,
      lineY: lineYs[index],
      labelY: labelYs[index],
    })),
  };
}

/**
 * Shorten a display name for the end label.
 *
 * The historical source formats names as "Sabalenka A", while the WTA API path
 * produces "Aryna Sabalenka". A trailing single letter is an initial, so the
 * surname is the leading token in that case and the trailing one otherwise.
 */
function shortName(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length < 2) return name;
  const last = parts[parts.length - 1];
  return last.length === 1 ? parts[0] : last;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
