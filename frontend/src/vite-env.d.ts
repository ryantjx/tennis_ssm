/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL the production bundle fetches `predictions.json` and
   * `results.json` from. Defaults to this repository's `outputs/latest`
   * directory on the `main` branch.
   */
  readonly VITE_DATA_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

