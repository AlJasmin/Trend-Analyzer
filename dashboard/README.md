# AI Con Dashboard

Polished, live dashboard for the Reddit AI discourse dataset.

## Quick start

1) Install dependencies:
   - `npm install`
2) Start the dev server:
   - `npm run dev`

The app reads MongoDB settings from `../config/settings.yaml` by default.
You can override with `.env` or environment variables.

## Environment overrides

- `MONGODB_URI`
- `MONGODB_DB`
- `DASHBOARD_SNAPSHOT_ONLY` (set to `1` to force snapshot mode)
- `DASHBOARD_TITLE`

## Snapshot fallback

Generate a static fallback file for offline demos:

```
npm run snapshot
```

This writes `dashboard/data/snapshot.json`. API routes will automatically use
the snapshot if MongoDB is unavailable (or if `DASHBOARD_SNAPSHOT_ONLY=1`).
