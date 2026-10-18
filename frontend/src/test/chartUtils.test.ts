import { describe, expect, it } from "vitest";
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
  nearestSamples,
  skillDomain,
  skillTicks,
} from "../chartUtils";

describe("chart scales", () => {
  it("maps a domain onto a pixel range", () => {
    const scale = createScale([0, 10], [0, 100]);

    expect(scale(0)).toBe(0);
    expect(scale(5)).toBe(50);
    expect(scale(10)).toBe(100);
  });

  it("supports an inverted range for screen coordinates", () => {
    const scale = createScale([-1, 1], [300, 0]);

    expect(scale(-1)).toBe(300);
    expect(scale(0)).toBe(150);
    expect(scale(1)).toBe(0);
  });

  it("centres rather than dividing by zero when the domain is empty", () => {
    const scale = createScale([2, 2], [0, 100]);

    expect(scale(2)).toBe(50);
  });
});

describe("date helpers", () => {
  it("round-trips an ISO date through epoch days", () => {
    expect(dayToDate(dateToDay("2026-10-17"))).toBe("2026-10-17");
    expect(dateToDay("1970-01-01")).toBe(0);
  });

  it("generates evenly spaced ticks that include both ends", () => {
    const ticks = dateTicks(dateToDay("2026-01-01"), dateToDay("2026-05-01"), 5);

    expect(ticks).toHaveLength(5);
    expect(ticks[0]).toBe(dateToDay("2026-01-01"));
    expect(ticks[ticks.length - 1]).toBe(dateToDay("2026-05-01"));
  });

  it("returns a single tick for a degenerate window", () => {
    expect(dateTicks(10, 10)).toEqual([10]);
  });
});

describe("series preparation", () => {
  it("keeps the last observation for each calendar day", () => {
    const collapsed = collapseByDate([
      { date: "2026-03-02", skill: 0.5 },
      { date: "2026-03-01", skill: 0.1 },
      { date: "2026-03-01", skill: 0.9 },
    ]);

    // Two matches on 2026-03-01 collapse to one point holding the later value.
    expect(collapsed).toEqual([
      { date: "2026-03-01", skill: 0.9 },
      { date: "2026-03-02", skill: 0.5 },
    ]);
  });

  it("filters points to an inclusive window", () => {
    const points = [
      { date: "2026-01-01", skill: 0.1 },
      { date: "2026-06-01", skill: 0.2 },
      { date: "2026-12-01", skill: 0.3 },
    ];

    const windowed = filterByWindow(
      points,
      dateToDay("2026-05-31"),
      dateToDay("2026-12-01"),
    );

    expect(windowed.map((point) => point.date)).toEqual(["2026-06-01", "2026-12-01"]);
  });

  it("builds a polyline path with one segment per point", () => {
    const x = createScale([0, 10], [0, 100]);
    const y = createScale([-1, 1], [100, 0]);

    const path = buildLinePath(
      [
        { date: dayToDate(0), skill: -1 },
        { date: dayToDate(10), skill: 1 },
      ],
      createScale([0, 10], [0, 100]),
      y,
    );

    expect(path.startsWith("M")).toBe(true);
    expect(path).toContain("L");
    // Two points means exactly one line command.
    expect(path.match(/L/g)).toHaveLength(1);
    expect(x(10)).toBe(100);
  });

  it("returns an empty path when there are no points", () => {
    expect(buildLinePath([], createScale([0, 1], [0, 1]), createScale([0, 1], [0, 1]))).toBe("");
  });

  it("breaks the line across a gap longer than the threshold", () => {
    const x = createScale([0, 200], [0, 200]);
    const y = createScale([0, 1], [100, 0]);
    const path = buildLinePath(
      [
        { date: dayToDate(0), skill: 0.2 },
        { date: dayToDate(10), skill: 0.3 },
        // 75 days later: no observations in between.
        { date: dayToDate(85), skill: 0.4 },
      ],
      x,
      y,
    );

    // Two move commands means two subpaths, so nothing is drawn across the gap.
    expect(path.match(/M/g)).toHaveLength(2);
    expect(path.match(/L/g)).toHaveLength(1);
  });

  it("keeps a single path when observations are dense", () => {
    const x = createScale([0, 200], [0, 200]);
    const y = createScale([0, 1], [100, 0]);
    const path = buildLinePath(
      [
        { date: dayToDate(0), skill: 0.2 },
        { date: dayToDate(7), skill: 0.3 },
        { date: dayToDate(14), skill: 0.4 },
      ],
      x,
      y,
    );

    expect(path.match(/M/g)).toHaveLength(1);
    expect(path.match(/L/g)).toHaveLength(2);
  });

  it("reports the gaps it breaks on", () => {
    const gaps = findGaps([
      { date: "2026-01-01", skill: 0.2 },
      { date: "2026-01-08", skill: 0.3 },
      { date: "2026-04-01", skill: 0.4 },
    ]);

    expect(gaps).toHaveLength(1);
    expect(gaps[0].from.date).toBe("2026-01-08");
    expect(gaps[0].to.date).toBe("2026-04-01");
    expect(gaps[0].days).toBeGreaterThan(DEFAULT_MAX_GAP_DAYS);
  });
});

describe("skill domain and ticks", () => {
  it("spans the data rather than pinning to zero", () => {
    const [min, max] = skillDomain([
      { points: [{ date: "2026-01-01", skill: 1 }] },
      { points: [{ date: "2026-01-02", skill: 2 }] },
    ]);

    // Zero is far outside this data, so including it would waste the lower half.
    expect(min).toBeCloseTo(0.9, 5);
    expect(max).toBeCloseTo(2.1, 5);
  });

  it("keeps zero on the axis when the data comes near it", () => {
    const [min, max] = skillDomain([
      { points: [{ date: "2026-01-01", skill: 0.05 }] },
      { points: [{ date: "2026-01-02", skill: 1 }] },
    ]);

    expect(min).toBeLessThanOrEqual(0);
    expect(max).toBeGreaterThan(1);
  });

  it("treats all-negative data the same way as all-positive data", () => {
    // Zero is as far from this data as it is in the all-positive case, so
    // pinning the axis to zero would waste the plot symmetrically.
    const [min, max] = skillDomain([
      { points: [{ date: "2026-01-01", skill: -1.5 }] },
      { points: [{ date: "2026-01-02", skill: -0.5 }] },
    ]);

    expect(min).toBeLessThan(-1.5);
    expect(max).toBeGreaterThan(-0.5);
    expect(max).toBeLessThan(0);
  });

  it("centres a single observation without collapsing the axis", () => {
    const [min, max] = skillDomain([{ points: [{ date: "2026-01-01", skill: 2 }] }]);

    expect(min).toBeLessThan(2);
    expect(max).toBeGreaterThan(2);
    expect(max - min).toBeGreaterThan(0);
  });

  it("is safe with no points", () => {
    expect(skillDomain([])).toEqual([-0.1, 0.1]);
  });

  it("produces a readable tick step", () => {
    const ticks = skillTicks([-1.2, 2.4]);

    expect(ticks.length).toBeGreaterThan(1);
    expect(ticks.length).toBeLessThan(9);
    // Ticks must be evenly spaced.
    const step = ticks[1] - ticks[0];
    for (let index = 2; index < ticks.length; index += 1) {
      expect(ticks[index] - ticks[index - 1]).toBeCloseTo(step, 8);
    }
  });
});

describe("hover resolution", () => {
  it("returns each player's nearest observation without interpolating", () => {
    const samples = nearestSamples(
      [
        {
          points: [
            { date: "2026-01-01", skill: 0.2 },
            { date: "2026-03-01", skill: 0.6 },
          ],
        },
      ],
      dateToDay("2026-02-20"),
    );

    expect(samples).toHaveLength(1);
    expect(samples[0].point).toEqual({ date: "2026-03-01", skill: 0.6 });
  });

  it("skips players with no observations", () => {
    expect(nearestSamples([{ points: [] }], dateToDay("2026-02-20"))).toEqual([]);
  });
});

describe("end label placement", () => {
  it("separates labels that would otherwise collide", () => {
    const placed = distributeLabels([100, 100.5, 101], 14, 0, 200);

    expect(placed[0]).toBe(100);
    expect(placed[1]).toBeGreaterThanOrEqual(placed[0] + 14);
    expect(placed[2]).toBeGreaterThanOrEqual(placed[1] + 14);
  });

  it("keeps labels inside the plot bounds", () => {
    const placed = distributeLabels([195, 197, 199], 14, 0, 200);

    for (const value of placed) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(200);
    }
    expect(placed[2]).toBe(200);
  });

  it("preserves the input order it was given", () => {
    const placed = distributeLabels([50, 10, 30], 14, 0, 100);

    // Sorted by position, then restored to input order.
    expect(placed).toHaveLength(3);
    expect(placed[1]).toBe(10);
  });

  it("returns an empty array for no labels", () => {
    expect(distributeLabels([], 14, 0, 100)).toEqual([]);
  });
});
