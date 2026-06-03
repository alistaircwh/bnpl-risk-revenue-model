# BNPL Merchant Risk & Revenue Prioritisation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PySpark](https://img.shields.io/badge/PySpark-3.3%2B-orange) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)

---

## The Problem

A BNPL firm has ~4,000 merchant candidates but capacity to onboard only the top 100. Historical revenue alone is a poor ranking signal: it's backwards-looking, ignores take rate, ignores stability, and crucially ignores fraud loss, which varies widely *within* revenue tiers and across consumer postcodes (8%–53%). This project combines forecasted revenue, take rate, and predicted fraud exposure into a single **risk-adjusted Expected Project Value (EPV)** score, then uses it to recommend a top-100 shortlist diversified across five industry segments.

---

## Pipeline

1. **ETL**: Clean and join 12.4M transactions, ~4,000 merchants, and 34,747 consumers; impute geographic features (LGA codes, income metrics) via KNN; integrate ABS demographic data
2. **Exploratory Analysis**: Geospatial fraud distribution, revenue mix by merchant segment, temporal trends
3. **Fraud Modelling**: Separate Random Forest regression models for consumer-level and merchant-level fraud probability
4. **Merchant Ranking**: Discounted Cash Flow (DCF) over a 3-period forecast horizon, weighted by combined fraud probability, to produce a risk-adjusted ranking across five industry segments

> Revenue forecasting was tested with a stacked LSTM (see [scripts/ranking_model_v2.py](scripts/ranking_model_v2.py)), but the model produced unstable forecasts across reruns and was not feasible to fine-tune across 3,000+ merchants. The final pipeline forecasts using the **average monthly revenue growth rate** over a 15-month window.

---

## Key Results

**Dataset scale:** 12.4M transactions · ~4,000 merchants · 34,747 consumers (3,212 merchants retained for ranking after filtering for 15 months of continuous sales)

### Fraud Models

| Model | RMSE | R² | Top Feature |
|-------|------|----|-------------|
| Consumer fraud (RFR) | 6.81 | 0.43 | Transaction dollar value (42.5% importance) |
| Merchant fraud (RFR) | 2.43 | **0.85** | Monthly order volume (31.4% importance) |

The merchant model is the stronger of the two; order volume and its month-to-month variance carry most of the signal. The consumer model is weaker because the only per-consumer signal beyond dollar value is the postcode-level fraud rate, and richer signals (transaction history length, category diversity, device/session) would likely push R² higher.

### Revenue by Merchant Tier

<p align="center">
  <img src="plots/boxplot_net_revenue.png" width="600" alt="Net revenue distribution across revenue levels a–e">
</p>

Level 'a' merchants account for **$47.8M** (~56% of total revenue) yet sit at the same fraud probability (~29%) as Level 'b' and 'c' merchants. Level 'e' merchants, by contrast, average **69% fraud probability** and contribute less than 0.1% of revenue, making them the clearest "do not partner" tier in the data.

| Revenue Level | Total Revenue | Avg Fraud Probability |
|---------------|---------------|----------------------|
| a | $47.8M | 29.5% |
| b | $28.0M | 31.6% |
| c | $8.97M | 29.8% |
| d | $0.33M | 63.4% |
| e | $0.06M | 69.1% |

### Geospatial Fraud Distribution

<p align="center">
  <img src="plots/average_fraud_prob_postcode.png" width="600" alt="Average fraud probability by postcode">
</p>

Consumer fraud probability ranges from **8% to 53% by postcode**, while state-level averages cluster tightly between **14.4%–15.5%**. Fraud is driven by local demographic factors, not state-wide patterns, so postcode-level (not state-level) averages were used as a feature in the consumer fraud model.

### Merchant Segments

<p align="center">
  <img src="plots/donut_chart_segments.png" width="480" alt="Merchant segment distribution">
</p>

The 4,026 merchant categories were grouped into five business segments. The top-100 shortlist is distributed equally across segments to avoid concentration risk on any single category.

<p align="center">
  <img src="plots/total_revenue_segments_v2.png" width="600" alt="Total estimated EPV by segment">
</p>

**Books, Media, Arts, Crafts, and Hobbies** leads at **$1.42M** total estimated EPV; **Home, Garden, and Furnishings** trails at $0.65M.

### Ranking Output

The top-ranked merchant by risk-adjusted EPV is *Dignissim Maecenas Foundation* (Fashion segment) at **~$54,713**. The top 10 merchants by commission each generate **>$490K** in estimated commission from **>$7.4M** in revenue. EPV is discounted by a combined fraud probability of `0.65 × P(merchant) + 0.35 × P(consumer)`. The asymmetry reflects loss magnitude per event (a merchant default wipes the entire forecasted stream; a fraudulent consumer transaction loses only that one transaction), not the relative importance of the two signals.

### Practical Outcomes

The final output is a ranked list of merchants ordered by risk-adjusted EPV. In practice, a BNPL platform could use this to:

- **Prioritise onboarding** for high-EPV, low-fraud merchants (Level 'a' band, ~29% fraud probability)
- **Flag for review** any merchant with fraud probability above ~50% before offering them partnership terms
- **Target marketing spend** toward the Books/Media and Computers/Electronics segments, which show the highest aggregate EPV
- **Deprioritise** Level 'e' merchants (avg 69% fraud probability, <0.1% of total revenue) until fraud controls are established

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ETL & ML pipelines | PySpark 3.3+, PySpark MLlib |
| Revenue forecasting (final) | PySpark aggregations (15-month growth rate, 3-period DCF) |
| Revenue forecasting (explored) | TensorFlow / Keras (stacked LSTM), rejected due to forecast instability |
| Geospatial imputation | scikit-learn (KNeighborsRegressor), geopandas |
| Visualisation | matplotlib, seaborn, folium (interactive maps) |
| Data wrangling | pandas, numpy |

---

## Project Structure

```
notebooks/          # Run in order 1 → 5
  1_ETL_pipeline              # Data cleaning, joining, KNN imputation
  2.1_preliminary_analysis    # Missing values, distributions, data quality
  2.2_geospatial_analysis     # Postcode/LGA choropleth maps
  2.3_visualisation           # Feature distributions, segment breakdowns
  3.1a/b_consumer_fraud_model # Consumer fraud probability (v1 → v2)
  3.2_merchant_fraud_model    # Merchant fraud probability
  4.1b_ranking_model_v2       # DCF forecasting + risk-adjusted ranking
  4.2_segments                # Segment profiling and top-merchant analysis
  5_summary                   # End-to-end review of findings

scripts/            # Reusable modules imported by notebooks
  etl_pipeline.py             # Core ETL functions and KNN imputation
  consumer_transaction_model.py  # Consumer fraud ML pipeline
  merchant_fraud.py           # Merchant fraud model with CrossValidator
  ranking_model_v2.py         # Growth-rate weight helpers + LSTM forecaster (exploratory)
  geospatial_analysis.py      # Folium map utilities
  visualisation.py            # Chart helpers
  preliminary_analysis.py     # Dataset profiling helpers
  consumer_model.py           # Consumer-level visualisation utilities

data/
  raw/              # Tracked: shapefiles, postcodes, ABS income/fraud data
  curated/          # Gitignored, generated by notebook 1
  tables/           # Gitignored, generated by notebook 1

plots/              # Saved visualisation outputs (PNG)
```

---

## Setup

**Prerequisites:** Python 3.8+, Java 8 or 11 (required by PySpark)

```bash
pip install -r requirements.txt
```

> **Note:** `data/curated/` and `data/tables/` are not tracked in git. Run `notebooks/1_ETL_pipeline.ipynb` first to generate them before running any subsequent notebooks.

**Run order:**
```
1_ETL_pipeline → 2.1 → 2.2 → 2.3 → 3.1a → 3.1b → 3.2 → 4.1b → 4.2 → 5_summary
```

For dataset schema details (merchant revenue bands, transaction structure, consumer ID mapping), see [`data/README.md`](data/README.md).

---

## Limitations & Future Work

- **Consumer fraud R² (0.43) is moderate.** Adding features such as consumer transaction history length, product category diversity, or device/session signals would likely improve prediction quality.
- **Ranking is limited to merchants with continuous sales history.** Only 3,212 of the ~4,000 merchants had 15 months of uninterrupted revenue and could be ranked; merchants with sparse history are excluded from the shortlist.
- **LSTM forecasting was rejected, not deployed.** The stacked LSTM in `ranking_model_v2.py` produced different forecasts on every rerun (random weight init + non-convex loss), and per-merchant fine-tuning across 3,000+ merchants was not computationally feasible within project scope.
- **Fraud labels are simulated.** The dataset uses a fraud delta file applied to synthetic transaction data, so real-world fraud distributions would differ in ways that could shift model behaviour significantly.
- **Fixed fraud weighting.** The 65% merchant / 35% consumer split was chosen heuristically (merchant default destroys the full revenue stream, consumer default is per-transaction); a sensitivity analysis across weighting combinations was not performed.
- **KNN imputation for geospatial features.** LGA codes and income metrics for postcodes with missing data are imputed using nearest-neighbour geographic proximity, which may not reflect actual socioeconomic conditions in low-density areas.

---

## Team

1. Do Nhat Anh Ha
2. Alistair Cheah Wern Hao
3. Sitao Qin
4. Shiping Song
