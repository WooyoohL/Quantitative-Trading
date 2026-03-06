# A-Quant-PyTorch: Deep Learning-Driven Quantitative Research and Trading System for A-Shares

**Operating Environment:** `D:\anaconda3\envs\qt`

**Project Vision:** This project aims to construct an end-to-end quantitative trading system based on **PyTorch**, specifically optimized for the **A-share market (T+1 settlement system)**. The system encompasses the full pipeline from automated data ingestion and deep feature engineering to SOTA model training and live execution (QMT compatible).

---

### Project Structure
Understand module dependencies based on this directory structure:

* **data/**: Data storage and processing
    * `fetcher.py`: Data crawling tool based on AkShare.
    * `processor.py`: Rolling normalization, outlier removal, and feature construction (PyTorch Dataset).
* **models/**: Model definitions
    * `encoders.py`: Transformer / LSTM / Gated-CNN backbones.
    * `loss_functions.py`: Rank-based Loss and Sharpe Loss specifically designed for quant finance.
* **strategy/**: Strategy layer
    * `alpha_generator.py`: Generates daily stock selection scores (Alpha Signals).
    * `backtest.py`: Local backtesting based on Backtrader or VectorBT.
* **execution/**: Execution layer
    * `mock_trader.py`: Simulated matching environment.
* **config.yaml**: Global hyperparameters (Lookback window, Universe, LR).
* **main.py**: Entry point for the full-process execution.

---

### Tech Stack
* **Data Source:** AkShare (A-share daily and minute lines, industry classifications).
* **Deep Learning:** PyTorch 2.x + HuggingFace Transformers (for sequence modeling).
* **Data Analysis:** Pandas, NumPy.
* **Backtesting Engines:** Backtrader (event-driven) or VectorBT (vectorized, suitable for high frequency).
* **Target Environment:** A-share Shanghai Stock Exchange (T+1 trading rules).

---

### Logical Design

#### 1. Data Pipeline
* **Input:** Open, High, Low, Close, Volume, and Turnover for the past $N$ days.
* **Features:** Automated calculation of technical indicators (MACD, RSI) + Deep feature learning.
* **Labels:** Predict the return rate of day $T+2$ relative to the opening price of day $T+1$ (to circumvent T+1 restrictions).

#### 2. Modeling
* **Objective Function:** Optimize the **Spearman Correlation (IC)** between predicted values and ground truth rankings.
* **Robustness:** Incorporate Dropout and LayerNorm to prevent overfitting to A-share market noise.

#### 3. Execution
* **Semi-Automatic Mode:** Update the model after the daily market close and output a `top_k.csv` selection list.

---

### Rolling Window Training Strategy
**Target Scenarios:** Ultra-short-term/Short-term trading (holding for 1–3 days), capturing intraday volatility and overnight premiums.

To ensure the model captures the rapid sector rotation and sentiment shifts in A-shares, the system adopts a daily/weekly rolling window:

* **Window Configuration:**
    * **Train Window:** Recommended use of the most recent 60–120 trading days. Ultra-short-term strategies do not require distant history (e.g., 3 years ago), as market ecology (e.g., quant proportion, regulatory environment) has changed drastically.
    * **Validation Window:** The most recent 10 trading days, used for hyperparameter fine-tuning.
    * **Test/Inference:** Predicting signals for the next 1–2 trading days.

* **Update Frequency:**
    * **Full-Auto Mode:** After each trading day close, automatically merge the day's transaction data into the Tensor, remove the oldest day, and trigger an incremental training (Fine-tuning) session.
    * **Semi-Auto Mode:** Perform a full retraining every weekend to recalibrate model weights.

---

### Retraining Trigger Monitoring Indicators
**Core Logic:** Trading must stop immediately and a forced retrain must be triggered when live performance deviates from the backtest distribution. The system monitors the following four core indicators, issuing an alert if any threshold is met:

1. **Information Coefficient (IC) Decay**
    * **Metric:** Rank_IC (Correlation between predicted rank and actual return rank).
    * **Trigger:** If the 5-day moving average Rank_IC remains below 0.02 (or is significantly lower than 50% of the backtest average).
    * **Implication:** The model can no longer identify high-potential stocks; predictions have become random guesses.

2. **Monotonicity Break (Group Return Inversion)**
    * **Metric:** Top_Group_Returns vs. Bottom_Group_Returns.
    * **Trigger:** Returns of the Top group (highest predicted scores) are lower than the market average or lower than the Bottom group for 3 consecutive days.
    * **Implication:** Market style has "reversed"; the logic the model originally favored (e.g., strong-stock breakouts) has become a losing strategy.

3. **Concept Drift Monitoring**
    * **Metric:** K-S Test or PSI (Population Stability Index) of feature distributions.
    * **Trigger:** PSI > 0.25 for current volume, volatility, or turnover distributions compared to the training set.
    * **Implication:** The market has entered an "abnormal state" (e.g., extreme volume contraction or expansion) that the old model has not encountered.

4. **Turnover & Cost Warning**
    * **Metric:** Realized_Slippage (Live trading slippage).
    * **Trigger:** Actual transaction costs exceed 40% of predicted profits.
    * **Implication:** For small-scale individual trading, fees and slippage are profit killers. If slippage is too high, the model must be retrained to reduce turnover frequency or optimize order execution algorithms.