# Khmer Tokenizer Fertility Benchmark — Website

Two independent parts:

- **`api/`** — FastAPI backend serving live tokenization for the demo section.
- **`web/`** — Vue 3 + Vite + TypeScript static frontend, deployed to GitHub Pages.

This folder does not touch or depend on the existing benchmark/corpus/training
code elsewhere in the repo — it only reads a static `results.json` for the
leaderboard and calls a small `/tokenize` endpoint for the live demo.

## Backend (`api/`)

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit paths/origins/model ids
uvicorn app.main:app --reload
```

Runs at `http://localhost:8000`. `GET /tokenizers` lists what actually loaded;
`POST /tokenize` runs the live demo.

**Before this fully works, edit `.env`:**
- `SP_BPE_32K_MODEL_PATH` / `SP_UNIGRAM_32K_MODEL_PATH` — point these at your
  actual local SentencePiece `.model` files.
- `SEALION_MODEL_ID` / `PRAHOKBART_MODEL_ID` — I was not fully certain of the
  current exact HuggingFace repo IDs for SEA-LION and PrahokBART at the time
  of writing, so verify/update these before relying on them. Any loader that
  fails (bad path, wrong model id, no network) is logged and skipped, not
  fatal — check the startup logs to see which tokenizers actually loaded.

### Docker (for Hugging Face Spaces)

```bash
cd api
docker build -t khmer-tokenizer-api .
docker run -p 7860:7860 --env-file .env khmer-tokenizer-api
```

## Frontend (`web/`)

```bash
cd web
npm install
cp .env.example .env   # set VITE_API_URL if not using localhost:8000
npm run dev
```

Runs at `http://localhost:5173`.

**Before deploying:**
- Update `REPO_NAME` in `vite.config.ts` to match your actual GitHub repo name
  (needed for the `/repo-name/` base path GitHub Pages project sites require).
- Replace the placeholder Paper/Code/Data links in `src/components/Header.vue`.
- Fill in `public/results.json` with your real benchmark numbers (currently
  dummy data).
- Fill in the three TODO error-taxonomy examples in
  `src/components/ErrorTaxonomy.vue`.
- In your repo's GitHub Settings → Pages, set the source to "GitHub Actions" —
  the included workflow (`.github/workflows/deploy.yml`) builds and deploys
  `web/` on every push to `main` that touches it.
- If your deployed API needs a non-default URL, set a repository *variable*
  named `VITE_API_URL` (Settings → Secrets and variables → Actions →
  Variables) so the deploy workflow bakes it into the build.

## Structure

```
api/
  app/
    main.py        FastAPI app, CORS, /tokenize and /tokenizers routes
    config.py       pydantic-settings, reads .env
    registry.py     tokenizer loader registry (loads once at startup)
    schemas.py      request/response models
  requirements.txt
  Dockerfile
  .env.example

web/
  src/
    components/
      Header.vue
      Leaderboard.vue      sorts a static public/results.json
      LiveDemo.vue         calls POST /tokenize
      ErrorTaxonomy.vue    static, TODO content
    api.ts
    types.ts
  public/results.json      placeholder leaderboard data
  vite.config.ts
  .env.example

.github/workflows/deploy.yml   builds web/ and deploys to GitHub Pages
```

## Not touched

Nothing under the rest of the repo (existing benchmark scripts, corpus files,
training code) was read or modified to build this — everything above is new,
self-contained, and lives only under `api/`, `web/`, and `.github/`.
