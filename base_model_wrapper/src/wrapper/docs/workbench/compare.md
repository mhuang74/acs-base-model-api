# Compare mode

**Compare mode** runs one prompt across several **lanes** at once — each lane its own model and sampling config — and streams the results side by side. It's the quick way to answer "how do these two models continue the same prompt?" or "what does `temperature` 0.2 vs 1.2 do here?".

Open it from any Workbench chat: flip the **Single / Compare** toggle in the composer. (The old `…/workbench/<chat>/compare` link still works — it just redirects you into the chat with Compare mode active.)

## 1. Set it up

1. Type your **prompt** once at the top — it's shared by every lane.
2. Each **lane** has its own **model** picker and full sampling config: max tokens, temperature, `top_p` / `top_k` / `min_p`, the presence / frequency / repetition penalties, `seed`, and `stop`.
3. **Duplicate** a lane to compare the same model at different settings, or add more lanes for more models (up to 6).

## 2. Run

Click **Run all** to fire every lane, or the per-lane **Run** to launch just one. Lanes stream **independently and side by side** — a slow or cold lane doesn't hold up the others. **Stop all** cancels everything in flight. Running compare does **not** touch your single-pane chat history.

## 3. Saved runs

Every **Run all** is saved automatically as a snapshot — no save button. A **compare-run history** list appears at the bottom of the pane, newest first — each row shows the timestamp, the shared prompt, and the lane count; expand a row to see the per-lane model chips. Click **Restore** to reload that run: the lanes, prompt, and every stored completion come straight back. The most recent 20 runs are kept per chat; older ones fall off.

## 4. Heatmap

Toggle **Heatmap** on a lane (and choose how many alternatives to record) to colour each token by probability — **green = confident, red = surprising** — using the same colour ramp as the [Loom](/tutorial/workbench/loom). **Hover** a token to see the top alternatives the model considered.

## 5. Billing

All lanes bill to one API key (the chat's key, or your default). Each lane is a **separate generation**, so a 4-lane run counts as 4 generations against your usage and budget.

## Tip: cold starts

Always-on models respond immediately. Putting an **on-demand large model** (e.g. 405B) in a lane triggers a cold boot for *that* lane — it will sit "warming up" for a few minutes before tokens appear, while your other lanes stream normally. See [Cold-boot waiting](/tutorial/examples/cold-boot).
