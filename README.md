# ACS base-model platform — take-home

This is a working copy of our base-model hosting API. The task description was
sent to you separately — this file is just how to get the thing running.

## What's here

- `base_model_wrapper/` — the API server (FastAPI). This is where the work is.
- `setup-dev.sh` — one command to run everything locally.
- `stub_upstream.py` — a fake model backend (returns placeholder text; there's
  no real GPU, and you don't need one).

## Requirements

- **Docker** running (for a throwaway local Postgres).
- **[uv](https://docs.astral.sh/uv/)** (Python package manager). Install:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Internet on first boot only (pulls the Postgres image + a public tokenizer).
  After that it runs offline.
- Ports **5173**, **5434** and **8900** free (see below if they aren't).

## Run it

```
./setup-dev.sh up
```

This starts the database, the fake backend, and the API server, and prints an
API key and a ready-to-run `curl`. The server listens on
`http://localhost:5173`. Leave it running; work in another terminal.

It also prints a **web-UI login** (email + password). The browser UI —
Workbench, Loom and Compare — is at `http://localhost:5173/`, and some of the
work here is in that UI rather than in the API, so it's worth logging in and
clicking around before you start. Override the credentials with `DEV_EMAIL` /
`DEV_PASSWORD` if you like.

Get a fresh API key any time, or re-print the login:

```
./setup-dev.sh key
./setup-dev.sh login
```

Tear down (stops the database and the fake backend):

```
./setup-dev.sh down
```

If startup ends with `address already in use`, something else owns one of the
ports. Run everything on your own set instead:

```
PG_CONTAINER=acs-devdb-2 PG_PORT=5435 STUB_PORT=8901 APP_PORT=5174 ./setup-dev.sh up
```

The fake backend honours `max_tokens`, returns a `logprobs` block when you ask
for one, and reports `usage` token counts; `STUB_MODE=drop_midway` and
`STUB_MODE=error_500` make it misbehave on purpose. Its log is at
`/tmp/acs-hiring-stub-<STUB_PORT>.log`. See the docstring in `stub_upstream.py`.

## Running the tests

Leave `./setup-dev.sh up` running and point the suite at the Postgres it
started:

```
cd base_model_wrapper
TEST_DATABASE_URL=postgresql://pg:pg@localhost:5434/acs \
  uv run --python 3.12 --extra dev pytest
```

That is the full run: **about 1059 passed**, in roughly three to five minutes.
(`pytest` lives in the `dev` extra, so `--extra dev` is what pulls it in on a
clean checkout. If you changed `PG_PORT`, use your port here too.)

Two things that are not your fault:

- **Re-run a whole file, not a single test.** A handful of tests depend on
  state their neighbours set up, so running one by node id can fail even
  though the file and the full suite are green — `test_harvest_api.py` has one
  like this. If you want to re-check something, run its file:
  `pytest tests/test_harvest_api.py`.
- A `RuntimeError: No active exception to reraise` printed *after* the summary
  line comes from a dependency's shutdown handler. The run already finished;
  the exit code is what counts.

`TEST_DATABASE_URL` is not optional. Without it, 450 database-backed tests skip
themselves **and 3 tests in `tests/test_startup_prewarm.py` fail** — they boot
the app's lifespan, which needs a database DSN to build its engine. That
failure is the missing variable, not a bug in the code. If you see exactly
those three red, you forgot the variable.

## Notes

- Token counts are approximate locally (we use a public tokenizer, not the
  model's real one).
