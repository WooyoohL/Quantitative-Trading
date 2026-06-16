# Paper Trading Cloud Workflow

## Purpose

This workflow runs a daily A-share paper-trading loop after market close.

Trading rule:

- Signal date: day `T` after close
- Buy date: next trading day `T+1`, at open
- Sell date: next trading day after buy, `T+2`, at open
- Final buy count: top 3 stocks after event filtering
- Initial cash: 100000
- Target exposure: 70% of available cash, split evenly across final buys
- Lot size: 100 shares

## Daily Stages

### 1. Prepare

Run:

```powershell
python main.py paper-trade --config paper_trading.yaml --stage prepare
```

The prepare stage:

- updates local EOD data;
- executes pending sells and buys using the current trade date open price;
- trains a new model on Monday and Wednesday nights, or whenever no source run exists;
- runs inference with the current source run;
- builds market heat candidates;
- merges model and heat candidates;
- writes the event-filter input CSV and decision template.

Key outputs:

- `outputs/paper_trading/latest_filter_input.csv`
- `outputs/paper_trading/latest_event_filter_decisions.csv`
- `outputs/paper_trading/state.json`
- `outputs/paper_trading/paper_account.xlsx`

### 2. Event Filter

Use `strategy/post_filter.md` to review candidates in:

```text
outputs/paper_trading/latest_filter_input.csv
```

Fill:

```text
outputs/paper_trading/latest_event_filter_decisions.csv
```

Only rows with `recommended_action=Keep` are eligible for the next buy plan.

### 3. Finalize

Run:

```powershell
python main.py paper-trade --config paper_trading.yaml --stage finalize
```

The finalize stage:

- reads event filter decisions;
- keeps only `Keep` rows;
- takes the first 3 rows by review rank;
- creates the next trading day buy plan;
- updates state and Excel workbook.

## Cloud Automation Notes

The cloud task should run `prepare`, perform the event filter using web research and `strategy/post_filter.md`, write the decision CSV, then run `finalize`.

The workflow is designed to start from a cloud workspace without local `outputs/runs`. If the configured source run is unavailable, the prepare stage trains a new source model before inference.
