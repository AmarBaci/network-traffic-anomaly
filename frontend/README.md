# Dashboard Frontend

React + Vite dashboard for the Network Traffic Anomaly project. It visualizes
model metrics, run history, and comparison plots served by the FastAPI backend
(`src/dashboard_api.py`).

## Development

```bash
npm install
npm run dev
```

The dev server runs at http://localhost:5173.

Start the backend in a separate terminal first (from the repo root):

```bash
python -m uvicorn nta.dashboard_api:app --reload
```

Use the **Refresh** button in the UI to rebuild the dashboard payload after
running new experiments.

## Scripts

- `npm run dev` — start the Vite dev server with HMR
- `npm run build` — production build to `dist/`
- `npm run preview` — preview the production build locally
- `npm run lint` — run ESLint

See the [top-level README](../README.md) for the full project and experiment
workflow.
