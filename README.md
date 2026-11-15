# Clothics – Smart Seasonal Fashion Store (Streamlit + ML)

**Problem:** A clothing store (T-shirts, jeans, trousers, lowers, track suits, underwear – 20 products)
wants seasonal and festival discounts (e.g. winter wear clearance in March, Diwali, New Year) and needs to know
*how many units will sell* and *which discount maximises profit*.

## What the app does
- **Shop** – 20 products with automatic seasonal/festival discounts, cart and checkout (demo).
- **Offer Calendar** – every rule and a category × month discount heatmap.
- **AI Pricing Lab** – Gradient Boosting demand model, discount-vs-units/profit curves, best discount, 12-month forecast.
- **Batch Predict** – upload a CSV (`category, season, price, cost, month, discount_pct`) and download predictions.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```
CSVs in `data/` (`products.csv`, `sales_history.csv`) are created automatically if missing.
Use the sidebar month selector to simulate March (winter clearance), October/November (Diwali), etc.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `data_loader.py` | Product catalogue, synthetic sales generator, CSV loading |
| `model.py` | Training, prediction, discount optimiser |
| `utils.py` | Offer calendar, discount rules, formatting, validation |

## Edit offers
Change the `OFFERS` list in `utils.py` (target = `all`, a season, or a category).
Replace `data/sales_history.csv` with real sales data (same columns) to train on your own store.
