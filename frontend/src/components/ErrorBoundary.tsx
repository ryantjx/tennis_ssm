import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * Render a readable message instead of a blank page if a payload is malformed.
 *
 * The dashboard reads deeply nested fields (model params, rankings, market
 * prices), so a partially written or schema-shifted JSON file previously
 * produced a white screen with no explanation.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Dashboard render failed", error, info.componentStack);
  }

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <main className="load-state">
        <strong>The forecast could not be displayed.</strong>
        <span>{error.message}</span>
        <span>This usually means the published prediction data is malformed.</span>
      </main>
    );
  }
}
