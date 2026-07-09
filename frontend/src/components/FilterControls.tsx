import type { MatchFilters, SortMode } from "../types";

interface FilterControlsProps {
  filters: MatchFilters;
  onChange: (filters: MatchFilters) => void;
  surfaces: string[];
}

export function FilterControls({ filters, onChange, surfaces }: FilterControlsProps) {
  function update(next: Partial<MatchFilters>) {
    onChange({ ...filters, ...next });
  }

  return (
    <section className="filters" aria-label="Prediction filters">
      <label className="filters__search">
        <span>Search</span>
        <input
          value={filters.query}
          onChange={(event) => update({ query: event.target.value })}
          placeholder="Player, tournament, location"
        />
      </label>
      <label>
        <span>Surface</span>
        <select value={filters.surface} onChange={(event) => update({ surface: event.target.value })}>
          <option value="">All surfaces</option>
          {surfaces.map((surface) => (
            <option key={surface} value={surface}>{surface}</option>
          ))}
        </select>
      </label>
      <label>
        <span>Sort</span>
        <select value={filters.sort} onChange={(event) => update({ sort: event.target.value as SortMode })}>
          <option value="date-desc">Newest first</option>
          <option value="date-asc">Oldest first</option>
          <option value="confidence-desc">Highest confidence</option>
          <option value="confidence-asc">Lowest confidence</option>
          <option value="edge-desc">Largest edge</option>
          <option value="log-score-desc">Best log score</option>
          <option value="log-score-asc">Worst log score</option>
        </select>
      </label>
    </section>
  );
}
