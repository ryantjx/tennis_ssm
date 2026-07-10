import { useMemo, useState } from "react";
import type { MatchPrediction } from "../types";
import { formatDate, matchKey } from "../utils";
import { PredictionCard } from "./PredictionCard";
import { PredictionListRow } from "./PredictionListRow";
import { ViewToggle } from "./UpcomingMatches";

interface CompletedResultsProps {
  matches: MatchPrediction[];
  resultWindow?: {
    start: string;
    end: string;
  };
  onOpen: (match: MatchPrediction) => void;
}

export function CompletedResults({ matches, resultWindow, onOpen }: CompletedResultsProps) {
  const [view, setView] = useState<"cards" | "list">("list");
  const sortedMatches = useMemo(
    () => [...matches].sort((left, right) =>
      right.timestamp - left.timestamp || matchKey(right).localeCompare(matchKey(left))),
    [matches],
  );

  return (
    <section className="section" id="completed" aria-labelledby="completed-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Forecast archive</span>
          <h2 id="completed-title">Completed results</h2>
        </div>
        <div className="section-heading__tools">
          <p>
            {resultWindow
              ? `Historical results from ${formatDate(resultWindow.start)} through ${formatDate(resultWindow.end)}.`
              : "Historical results are stored locally for reference."}
          </p>
          <ViewToggle view={view} onChange={setView} label="Completed view" />
        </div>
      </div>

      {matches.length ? (
        <div className={view === "cards" ? "match-grid completed-scroll" : "match-list completed-scroll"}>
          {sortedMatches.map((match) =>
            view === "cards"
              ? <PredictionCard key={matchKey(match)} match={match} onOpen={onOpen} />
              : <PredictionListRow key={matchKey(match)} match={match} onOpen={onOpen} />
          )}
        </div>
      ) : (
        <div className="empty-state"><strong>No completed forecast rows match the filters.</strong></div>
      )}
      {matches.length ? (
        <div className="list-progress">
          <p aria-live="polite">Showing {sortedMatches.length} of {matches.length} matches</p>
        </div>
      ) : null}
    </section>
  );
}
