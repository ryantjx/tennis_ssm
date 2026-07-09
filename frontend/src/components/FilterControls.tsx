import type { MatchFilters } from "../types";

interface FilterControlsProps {
  filters: MatchFilters;
  onChange: (filters: MatchFilters) => void;
  resultCount: number;
  totalCount: number;
}

export function FilterControls({ filters, onChange, resultCount, totalCount }: FilterControlsProps) {
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
      <p className="filters__count">{resultCount} of {totalCount} matches</p>
    </section>
  );
}
