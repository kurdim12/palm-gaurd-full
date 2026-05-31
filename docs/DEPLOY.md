# Palm Guard — Deployment

Reference deploy: **API** on Render/Railway, **frontend** on Vercel, **DB** on
Supabase. All services read configuration from environment variables documented
in `.env.example`.

## 1. Database — Supabase

1. Create a Supabase project.
2. Apply the schema (SQL editor, or `psql`):
   ```bash
   psql "$SUPABASE_DB_URL" -f packages/api/app/db/schema.sql
   ```
3. Note the project URL and the **service role** key (server-side only — never
   ship it to the browser).

## 2. Backend API — Render / Railway

The API is a standard ASGI app (`packages/api/app/main.py`).

**Start command:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
**Root / build:** `packages/api`, `pip install -r requirements.txt` (uncomment
`supabase` and, for live alerts, `twilio`).

**Environment:**
| Var | Purpose |
|---|---|
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | live DB (omit → in-memory) |
| `STATUS_INFESTED_STREAK`, `STATUS_CONFIDENCE_THRESHOLD` | debounce tuning |
| `ALERT_PROVIDER` | `log` \| `twilio` \| `whatsapp` \| `none` |
| `ALERT_RATE_LIMIT_MINUTES` | alert dedupe window |
| `TWILIO_ACCOUNT_SID`/`TWILIO_AUTH_TOKEN`/`TWILIO_FROM`/`ALERT_TO` | provider creds |

`render.yaml` and `railway.json` at the repo root capture this.

## 3. Frontend — Vercel

- **Root directory:** `packages/web`
- **Framework preset:** Next.js (build `next build`, output auto-detected)
- **Environment:** `NEXT_PUBLIC_API_URL=https://<your-api-host>`

`packages/web/vercel.json` pins the root directory.

## 4. Edge devices

Not "deployed" centrally — flashed per device. See `docs/HARDWARE.md`. Point each
device's `EDGE_API_URL` at the deployed API.

## 5. Smoke test after deploy

```bash
curl https://<api-host>/health
# {"status":"ok","db":"supabase","version":"0.1.0"}

# End-to-end (also runnable locally): see scripts/demo.sh
```
