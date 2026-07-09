import { useState } from "react";
import type { CompletedResult, MatchPrediction } from "../types";
import { formatDate } from "../utils";
import { PredictionCard } from "./PredictionCard";
import { PredictionListRow } from "./PredictionListRow";
import { ViewToggle } from "./UpcomingMatches";

interface CompletedResultsProps {
  matches: MatchPrediction[];
  results: CompletedResult[];
  resultWindow?: {
    start: string;
    end: string;
  };
  onOpen: (match: MatchPrediction) => void;
}

export function CompletedResults({ matches, resultWindow, onOpen }: CompletedResultsProps) {
  const [view, setView] = useState<"cards" | "list">("list");

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
          {matches.map((match) =>
            view === "cards"
              ? <PredictionCard key={match.id ?? `${match.date}-${match.player1}-${match.player2}`} match={match} onOpen={onOpen} />
              : <PredictionListRow key={match.id ?? `${match.date}-${match.player1}-${match.player2}`} match={match} onOpen={onOpen} />
          )}
        </div>
      ) : (
        <div className="empty-state"><strong>No completed forecast rows match the filters.</strong></div>
      )}
    </section>
  );
}
