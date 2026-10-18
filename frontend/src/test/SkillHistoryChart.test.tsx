import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { LABEL_MIN_GAP } from "../chartUtils";
import { SkillHistoryChart } from "../components/SkillHistoryChart";
import type { SkillHistory } from "../types";

const history: SkillHistory = {
  window: { start: "2025-10-18", end: "2026-10-17", days: 365 },
  players: [
    {
      id: 0,
      name: "Aryna Sabalenka",
      rank: 1,
      points: [
        { date: "2026-01-19", skill: 1.4 },
        { date: "2026-06-29", skill: 1.9 },
        { date: "2026-09-01", skill: 1.7 },
      ],
    },
    {
      id: 1,
      name: "Iga Swiatek",
      rank: 2,
      points: [
        { date: "2026-01-19", skill: 1.1 },
        { date: "2026-06-29", skill: 1.2 },
        { date: "2026-09-01", skill: 1.35 },
      ],
    },
    {
      id: 2,
      name: "Coco Gauff",
      rank: 3,
      points: [
        { date: "2026-01-19", skill: 0.9 },
        { date: "2026-09-01", skill: 1.05 },
      ],
    },
  ],
  slams: [
    {
      key: "aus-open",
      name: "Australian Open",
      short: "AO",
      surface: "Hard",
      start: "2026-01-18",
      end: "2026-01-31",
    },
    {
      key: "french-open",
      name: "French Open",
      short: "RG",
      surface: "Clay",
      start: "2026-05-24",
      end: "2026-06-07",
    },
    {
      key: "wimbledon",
      name: "Wimbledon",
      short: "WIM",
      surface: "Grass",
      start: "2026-06-29",
      end: "2026-07-12",
    },
    {
      key: "us-open",
      name: "US Open",
      short: "USO",
      surface: "Hard",
      start: "2026-08-31",
      end: "2026-09-13",
    },
  ],
};

afterEach(cleanup);

describe("SkillHistoryChart", () => {
  it("draws one path per ranked player", () => {
    const { container } = render(<SkillHistoryChart history={history} />);

    const lines = container.querySelectorAll("path.skill-chart__line");
    expect(lines).toHaveLength(3);
    expect(lines[0].getAttribute("data-player")).toBe("Aryna Sabalenka");
    expect(lines[0].getAttribute("d")).toContain("M");
  });

  it("labels all four Grand Slam bands", () => {
    const { container } = render(<SkillHistoryChart history={history} />);

    const labels = [...container.querySelectorAll(".skill-chart__slam-label")].map(
      (node) => node.textContent,
    );
    expect(labels).toEqual(["AO", "RG", "WIM", "USO"]);
  });

  it("names each Slam for assistive technology", () => {
    render(<SkillHistoryChart history={history} />);

    expect(screen.getByLabelText(/Australian Open, 2026-01-18 to 2026-01-31/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Wimbledon, 2026-06-29 to 2026-07-12/)).toBeInTheDocument();
  });

  it("lists players in rank order in the legend", () => {
    render(<SkillHistoryChart history={history} />);

    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(3);
    expect(buttons[0]).toHaveAccessibleName(/Aryna Sabalenka/);
    expect(buttons[0]).toHaveAccessibleName(/#1/);
    expect(buttons[2]).toHaveAccessibleName(/Coco Gauff/);
  });

  it("toggles emphasis when a legend item is selected", () => {
    render(<SkillHistoryChart history={history} />);

    const button = screen.getByRole("button", { name: /Aryna Sabalenka/ });
    expect(button).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-pressed", "true");
  });

  it("does not interpolate values between observations on hover", () => {
    const { container } = render(<SkillHistoryChart history={history} />);
    const svg = container.querySelector("svg.skill-chart__svg") as SVGSVGElement;

    // jsdom has no layout, so the bounding box is stubbed to give the scale a
    // real pixel width. Halfway across the plot resolves near mid-window.
    svg.getBoundingClientRect = () => ({
      x: 0,
      y: 0,
      width: 900,
      height: 380,
      top: 0,
      left: 0,
      right: 900,
      bottom: 380,
      toJSON: () => ({}),
    }) as DOMRect;

    fireEvent.mouseMove(svg, { clientX: 450 });

    const table = screen.getByRole("table", { name: "Skill at hovered date" });
    // Reported rows are real observations, never values between two dates.
    expect(within(table).getByText("Aryna Sabalenka")).toBeInTheDocument();
    expect(within(table).getAllByText(/^2026-/).length).toBeGreaterThan(0);
    for (const cell of within(table).getAllByText(/^2026-/)) {
      expect(history.players.flatMap((player) => player.points.map((p) => p.date)))
        .toContain(cell.textContent);
    }
  });

  it("renders an empty state when history is missing", () => {
    render(<SkillHistoryChart />);

    expect(screen.getByText("No skill history is available yet.")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("renders an empty state when no player has points in the window", () => {
    render(
      <SkillHistoryChart
        history={{ ...history, players: [{ ...history.players[0], points: [] }] }}
      />,
    );

    expect(screen.getByText("No skill history is available yet.")).toBeInTheDocument();
  });

  it("orders end labels by where the lines finish, not by rank", () => {
    // Rank 3 finishes above ranks 1 and 2, so its label must sit highest.
    // Stacking labels in rank order here would make the leader lines cross.
    const { container } = render(
      <SkillHistoryChart
        history={{
          ...history,
          players: [
            { id: 0, name: "Player One", rank: 1, points: [{ date: "2026-09-01", skill: 0.2 }] },
            { id: 1, name: "Player Two", rank: 2, points: [{ date: "2026-09-01", skill: 0.5 }] },
            { id: 2, name: "Player Three", rank: 3, points: [{ date: "2026-09-01", skill: 2.2 }] },
          ],
        }}
      />,
    );

    const labels = [...container.querySelectorAll(".skill-chart__end-label")].map((group) => ({
      name: group.querySelector("text")!.textContent!,
      lineY: Number(group.querySelector("line")!.getAttribute("y1")),
      labelY: Number(group.querySelector("text")!.getAttribute("y")),
    }));

    // Highest skill has the smallest y (top of the plot). End labels show the
    // surname only, so "Player Three" is labelled "#3 Three".
    const highest = labels.find((label) => label.name?.includes("Three"))!;
    expect(highest.labelY).toBeLessThan(
      labels.find((label) => label.name?.includes("One"))!.labelY,
    );

    // Every label must sit at or above the line that follows it, and no leader
    // may stretch further than clearing the neighbouring label requires.
    for (const label of labels) {
      expect(Math.abs(label.labelY - label.lineY)).toBeLessThanOrEqual(3 * LABEL_MIN_GAP);
    }
    // Labels stacked in visual order means their y values are monotonic with the
    // line ends they point at.
    const sorted = [...labels].sort((left, right) => left.lineY - right.lineY);
    for (let index = 1; index < sorted.length; index += 1) {
      expect(sorted[index].labelY).toBeGreaterThanOrEqual(sorted[index - 1].labelY);
    }
  });

  it("marks a player whose last observation is stale", () => {
    // Window ends 2026-10-17; this player last played 2026-09-05.
    const { container } = render(
      <SkillHistoryChart
        history={{
          ...history,
          players: [
            {
              id: 0,
              name: "Lapsed Player",
              rank: 1,
              points: [
                { date: "2026-01-19", skill: 1.0 },
                { date: "2026-09-05", skill: 0.9 },
              ],
            },
          ],
        }}
      />,
    );

    const stale = container.querySelectorAll("circle.skill-chart__stale");
    expect(stale).toHaveLength(1);
    expect(stale[0].querySelector("title")?.textContent).toContain("2026-09-05");
  });

  it("does not mark a player who played recently", () => {
    const { container } = render(
      <SkillHistoryChart
        history={{
          ...history,
          players: [
            {
              id: 0,
              name: "Active Player",
              rank: 1,
              points: [
                { date: "2026-01-19", skill: 1.0 },
                { date: "2026-10-10", skill: 1.1 },
              ],
            },
          ],
        }}
      />,
    );

    expect(container.querySelectorAll("circle.skill-chart__stale")).toHaveLength(0);
  });

  it("labels the skill axis so the measure is defined on the chart itself", () => {
    render(<SkillHistoryChart history={history} />);

    expect(screen.getByText("Model skill, before each match")).toBeInTheDocument();
  });

  it("draws a dashed connector across a long gap", () => {
    const { container } = render(
      <SkillHistoryChart
        history={{
          ...history,
          players: [
            {
              id: 0,
              name: "Gap Player",
              rank: 1,
              points: [
                { date: "2025-11-01", skill: 0.5 },
                { date: "2026-04-01", skill: 0.6 },
              ],
            },
          ],
        }}
      />,
    );

    const paths = container.querySelectorAll("path.skill-chart__line");
    expect(paths[0].getAttribute("d")?.match(/M/g)).toHaveLength(2);
    expect(container.querySelectorAll(".skill-chart__gap")).toHaveLength(1);
  });

  it("drops points outside the declared window", () => {
    const { container } = render(
      <SkillHistoryChart
        history={{
          ...history,
          players: [
            {
              ...history.players[0],
              points: [
                { date: "2020-01-01", skill: -3 },
                { date: "2026-06-29", skill: 1.9 },
              ],
            },
            history.players[1],
          ],
        }}
      />,
    );

    expect(container.querySelectorAll("path.skill-chart__line")).toHaveLength(2);
  });
});
