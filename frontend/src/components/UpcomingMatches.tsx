import { useState } from "react";
import type { MatchPrediction, PredictionPayload } from "../types";
import { matchKey } from "../utils";
import { PredictionCard } from "./PredictionCard";
import { PredictionListRow } from "./PredictionListRow";

interface UpcomingMatchesProps {
  matches: MatchPrediction[];
  fixtureStatus?: PredictionPayload["fixture_status"];
  onOpen: (match: MatchPrediction) => void;
}

export function UpcomingMatches({ matches, fixtureStatus, onOpen }: UpcomingMatchesProps) {
  const [view, setView] = useState<"cards" | "list">("cards");

  return (
    <section className="section" id="upcoming" aria-labelledby="upcoming-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Next matches</span>
          <h2 id="upcoming-title">Upcoming predictions</h2>
        </div>
        <div className="section-heading__tools">
          <p>{fixtureStatusText(fixtureStatus)}</p>
          <ViewToggle view={view} onChange={setView} label="Upcoming view" />
        </div>
      </div>
      {matches.length ? (
        <div className={view === "cards" ? "match-grid" : "match-list"}>
          {matches.map((match) =>
            view === "cards"
              ? <PredictionCard key={matchKey(match)} match={match} onOpen={onOpen} />
              : <PredictionListRow key={matchKey(match)} match={match} onOpen={onOpen} />
          )}
        </div>
      ) : (
        <div className="empty-state">
          <strong>No matched future fixtures.</strong>
          <span>The next model run will check the WTA schedule again.</span>
        </div>
      )}
    </section>
  );
}

export function ViewToggle({
  view,
  onChange,
  label,
}: {
  view: "cards" | "list";
  onChange: (view: "cards" | "list") => void;
  label: string;
}) {
  return (
    <div className="view-toggle" role="group" aria-label={label}>
      <button type="button" aria-pressed={view === "cards"} onClick={() => onChange("cards")}>Cards</button>
      <button type="button" aria-pressed={view === "list"} onClick={() => onChange("list")}>List</button>
    </div>
  );
}

function fixtureStatusText(status?: PredictionPayload["fixture_status"]): string {
  if (!status) return "Future fixtures load from the WTA schedule.";
  if (status.error) return `WTA fixture source unavailable: ${status.error}`;
  const loaded = status.loaded ?? 0;
  const matched = status.matched_model_players ?? 0;
  return `${matched} of ${loaded} WTA fixtures matched trained model players.`;
}
