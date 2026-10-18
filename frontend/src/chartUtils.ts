import type { SkillHistoryPlayer, SkillHistoryPoint } from "./types";

const MS_PER_DAY = 86_400_000;

/** Smallest gap allowed between two end labels, in pixels. */
export const LABEL_MIN_GAP = 13;

export interface Scale {
  /** Maps a data value to a pixel coordinate. */
  (value: number): number;
  domain: [number, number];
  range: [number, number];
}

export function createScale(domain: [number, number], range: [number, number]): Scale {
  const [domainStart, domainEnd] = domain;
  const [rangeStart, rangeEnd] = range;
  const span = domainEnd - domainStart;
  const scale = ((value: number) => {
    // A degenerate domain would divide by zero; centre the value instead.
    if (span === 0 || !Number.isFinite(span)) return (rangeStart + rangeEnd) / 2;
    return rangeStart + ((value - domainStart) / span) * (rangeEnd - rangeStart);
  }) as Scale;
  scale.domain = domain;
  scale.range = range;
  return scale;
}

/** Convert an ISO date to whole days since the Unix epoch (UTC). */
export function dateToDay(date: string): number {
  return Math.round(Date.parse(`${date}T00:00:00Z`) / MS_PER_DAY);
}

/** Convert whole days since the Unix epoch back to an ISO date. */
export function dayToDate(day: number): string {
  if (!Number.isFinite(day)) return "";
  return new Date(day * MS_PER_DAY).toISOString().slice(0, 10);
}

/**
 * Keep one point per calendar day, retaining the last observation.
 *
 * The underlying series is per match, so a player with several matches on the
 * same day would otherwise produce a vertical step at a single x value. Keeping
 * the final value for the day gives a single-valued series for the date.
 */
export function collapseByDate(points: SkillHistoryPoint[]): SkillHistoryPoint[] {
  const byDate = new Map<string, SkillHistoryPoint>();
  for (const point of points) {
    byDate.set(point.date, point);
  }
  return [...byDate.values()].sort((left, right) => dateToDay(left.date) - dateToDay(right.date));
}

/** Restrict points to an inclusive day window. */
export function filterByWindow(
  points: SkillHistoryPoint[],
  startDay: number,
  endDay: number,
): SkillHistoryPoint[] {
  return points.filter((point) => {
    const day = dateToDay(point.date);
    return day >= startDay && day <= endDay;
  });
}

/** A break longer than this is drawn as a discontinuity, not a trend. */
export const DEFAULT_MAX_GAP_DAYS = 45;

/**
 * Compute the skill axis domain across the visible players.
 *
 * The domain is driven by the data rather than pinned to zero. Anchoring at zero
 * is correct for a chart that must show sign, but when every top-ten skill sits
 * between roughly 1 and 3.5 it collapses the lines into the top half of the plot.
 * Zero is therefore included only when the data comes close to it, so the sign is
 * still visible whenever it is meaningful.
 *
 * Percentile clipping is deliberately avoided: the OU process has a multi-year
 * half-life, so a genuine drift would be hidden rather than merely tidied.
 */
export function skillDomain(
  players: Array<Pick<SkillHistoryPlayer, "points">>,
  padRatio = 0.1,
): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const player of players) {
    for (const point of player.points) {
      if (!Number.isFinite(point.skill)) continue;
      min = Math.min(min, point.skill);
      max = Math.max(max, point.skill);
    }
  }

  // No usable points: fall back to a small symmetric domain around zero.
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [-0.1, 0.1];

  const span = max - min;
  // Include zero when the data already sits within a tenth of its span of it,
  // which keeps the sign readable without wasting the whole lower plot.
  if (span > 0 && Math.abs(min) <= span * padRatio && min > 0) min = 0;
  if (span > 0 && Math.abs(max) <= span * padRatio && max < 0) max = 0;

  const paddedSpan = max - min;
  const pad = paddedSpan === 0 ? Math.max(Math.abs(max), 1) * padRatio : paddedSpan * padRatio;
  return [min - pad, max + pad];
}

/**
 * Build an SVG path, breaking the line across long gaps in the data.
 *
 * The series is per match, so a player can go two months without an observation.
 * A single continuous line would draw a straight trend through that silence and
 * imply the model tracked a decline or rise it never observed. Segments longer
 * than ``maxGapDays`` therefore start a new subpath.
 */
export function buildLinePath(
  points: SkillHistoryPoint[],
  x: Scale,
  y: Scale,
  maxGapDays = DEFAULT_MAX_GAP_DAYS,
): string {
  if (!points.length) return "";
  const segments: string[] = [];
  let previousDay: number | null = null;

  for (const point of points) {
    const day = dateToDay(point.date);
    const broken = previousDay !== null && Math.abs(day - previousDay) > maxGapDays;
    const command = previousDay === null || broken ? "M" : "L";
    segments.push(`${command}${x(day).toFixed(2)},${y(point.skill).toFixed(2)}`);
    previousDay = day;
  }
  return segments.join(" ");
}

/**
 * Locate the boundaries of long gaps so the chart can mark them.
 *
 * Returns the pair of points on each side of every break, letting the renderer
 * draw a faint dashed connector that shows a hiatus rather than a trend.
 */
export function findGaps(
  points: SkillHistoryPoint[],
  maxGapDays = DEFAULT_MAX_GAP_DAYS,
): Array<{ from: SkillHistoryPoint; to: SkillHistoryPoint; days: number }> {
  const gaps: Array<{ from: SkillHistoryPoint; to: SkillHistoryPoint; days: number }> = [];
  for (let index = 1; index < points.length; index += 1) {
    const days = Math.abs(dateToDay(points[index].date) - dateToDay(points[index - 1].date));
    if (days > maxGapDays) {
      gaps.push({ from: points[index - 1], to: points[index], days });
    }
  }
  return gaps;
}

export interface NearestSample {
  playerIndex: number;
  point: SkillHistoryPoint;
}

/**
 * Find the sample each player held at a given day.
 *
 * Hovering is resolved against the observation dates rather than by interpolating
 * between them, so the readout never reports a value the model did not produce.
 */
export function nearestSamples(
  players: Array<Pick<SkillHistoryPlayer, "points">>,
  targetDay: number,
): NearestSample[] {
  const samples: NearestSample[] = [];
  players.forEach((player, playerIndex) => {
    let best: SkillHistoryPoint | null = null;
    let bestDistance = Infinity;
    for (const point of player.points) {
      const distance = Math.abs(dateToDay(point.date) - targetDay);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = point;
      }
    }
    if (best) samples.push({ playerIndex, point: best });
  });
  return samples;
}

/**
 * Spread label positions so they do not overlap.
 *
 * Desired positions arrive roughly ordered by value, which on an inverted chart
 * differs from visual order, so the sweep sorts first and restores the original
 * order on return.
 */
export function distributeLabels(
  desired: number[],
  minGap: number,
  min: number,
  max: number,
): number[] {
  if (!desired.length) return [];

  const order = desired
    .map((value, index) => ({ value, index }))
    .sort((left, right) => left.value - right.value);

  // Sweep downward, pushing each label past its predecessor.
  for (let index = 1; index < order.length; index += 1) {
    const minimum = order[index - 1].value + minGap;
    if (order[index].value < minimum) order[index].value = minimum;
  }

  // If the last label overflowed the plot, push the stack back up and re-sweep.
  const overflow = order[order.length - 1].value - max;
  if (overflow > 0) {
    for (const entry of order) entry.value -= overflow;
    for (let index = order.length - 1; index > 0; index -= 1) {
      const maximum = order[index].value - minGap;
      if (order[index - 1].value > maximum) order[index - 1].value = maximum;
    }
  }

  // Clamp inside the plot without disturbing relative order.
  for (let index = 0; index < order.length; index += 1) {
    order[index].value = Math.min(max, Math.max(min, order[index].value));
  }

  const result = new Array<number>(desired.length);
  for (const entry of order) result[entry.index] = entry.value;
  return result;
}

/** Choose roughly `count` evenly spaced date ticks inside the window. */
export function dateTicks(startDay: number, endDay: number, count = 5): number[] {
  if (endDay <= startDay) return [startDay];
  const step = (endDay - startDay) / Math.max(count - 1, 1);
  return Array.from({ length: count }, (_, index) => Math.round(startDay + step * index));
}

/** Choose a readable step for the skill axis. */
export function skillTicks(domain: [number, number], target = 5): number[] {
  const [min, max] = domain;
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) return [min];
  const rough = span / target;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const candidates = [1, 2, 2.5, 5, 10].map((factor) => factor * magnitude);
  const step = candidates.find((candidate) => candidate >= rough) ?? magnitude * 10;
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let value = first; value <= max + step * 1e-9; value += step) {
    // Guard against float drift producing values like 0.30000000000000004.
    ticks.push(Math.round(value / step) * step);
  }
  return ticks;
}

/** Format an axis date compactly for the tick row. */
export function formatAxisDate(day: number): string {
  return new Intl.DateTimeFormat("en-GB", {
    month: "short",
    year: "2-digit",
    timeZone: "UTC",
  }).format(new Date(day * MS_PER_DAY));
}
