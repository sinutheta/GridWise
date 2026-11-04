TASK: Replace the synthetic dataset with REAL UK carbon intensity data, then re-baseline.

CONTEXT
The current data/processed/intensity.parquet is synthetic (only 0.36% whole numbers,
perfect 1-year range, starts 2023-01-01). Do not use it for results. Keep it only as
a test fixture, renamed to data/processed/intensity_synthetic.parquet. All earlier
simulation numbers (oracle 12.69%, last-week 7.13%, yesterday 6.17%) are invalid.

STEP 1: Test the API with one request
- Fetch one day from the National Energy System Operator Carbon Intensity API
  (https://api.carbonintensity.org.uk) and print the raw JSON structure.
- Use the "actual" intensity field as the target, not "forecast".
- Confirm the exact endpoint format from the official API docs. Do not guess it.
- If the request fails, print the full error and the HTTP status code, and stop.3

STEP 2: Full pull
- Fetch national half-hourly data for 2022-01-01 up to the latest complete day.
- The API caps history at 14 days per request, so loop in 14-day chunks with a short
  pause, retries and exponential backoff.
- Cache each chunk to data/raw/ so a failed run can resume without refetching.
- Add data/raw/ and checkpoints/ to .gitignore if not already there.

STEP 3: Clean
- UTC timestamps, regular 30-minute index, drop duplicates (keep last).
- Linearly interpolate gaps of at most 2 slots and set is_imputed=True.
- Mark longer gaps is_valid=False and make sure windows crossing them are excluded
  downstream.
- Clip negative or impossible values and log how many.
- Save to data/processed/intensity.parquet.
- Print a data report: date range, row count, missing slots, imputed count,
  invalid count, share of whole numbers (should be near 1.0), mean/std/min/max.

STEP 4: Visual check
- Save results/figures/raw_two_weeks.png (two-week plot) and
  results/figures/full_series.png. Real data should look irregular and noisy,
  not a clean repeating wave.

STEP 5: Weather features
- Add historical weather (wind speed, cloud cover, temperature, solar radiation)
  from a free source such as Open-Meteo. Verify the licence and record the exact
  source in the README.
- Align to the 30-minute index (forward-fill hourly values).
- Add a config flag future_weather (default false) because observed future weather
  is optimistic. State this assumption in the README.

STEP 6: Splits
- Chronological split: about 70% train, 15% validation, 15% test, with a 48-hour
  gap between splits.
- Fit scalers on train only. Save them alongside the data.
- Print the date range of each split and confirm no overlap.

STEP 7: Re-baseline
- Re-run python -m src.simulate on the real test split for run_now, persistence,
  yesterday, last-week and oracle.
- Save main.csv, by_slack.csv and by_duration.csv.
- Sanity checks that must pass and be printed:
  * run_now has exactly 0% savings and 0 regret
  * oracle regret is never negative
  * savings increase as slack increases
  * baselines lie between run_now and oracle

STEP 8: Tests and commit
- Run python -m pytest tests/ -v and make sure all tests pass.
- Add tests for: no duplicate timestamps, regular 30-minute index, split gaps,
  imputed flag correctness.
- Commit with a clear message and push to origin main.

DO NOT
- Do not train any model yet.
- Do not overwrite the synthetic file without renaming it first.
- Do not commit data/raw/ or API keys.

DELIVERABLE (report back with)
1. The data report from Step 3.
2. The split date ranges from Step 6.
3. The new baseline table from Step 7.
4. Any errors, assumptions, or things you could not verify.