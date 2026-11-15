"""Load / create the product catalogue and (synthetic) sales history CSVs."""
from pathlib import Path

import numpy as np
import pandas as pd

from utils import MONTHS, get_discount, apply_discount

DATA_DIR = Path(__file__).parent / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
SALES_CSV = DATA_DIR / "sales_history.csv"

# (name, category, price, cost, season, emoji)
_CATALOGUE = [
    ("Classic Round-Neck T-Shirt", "T-Shirt",   499,  220, "all",    "👕"),
    ("Graphic Print T-Shirt",      "T-Shirt",   699,  300, "summer", "🎽"),
    ("Cotton Polo T-Shirt",        "T-Shirt",   899,  400, "all",    "👕"),
    ("Oversized Street Tee",       "T-Shirt",   799,  340, "all",    "👕"),
    ("Slim Fit Blue Jeans",        "Jeans",    1799,  800, "all",    "👖"),
    ("Straight Fit Jeans",         "Jeans",    1699,  750, "all",    "👖"),
    ("Distressed Denim Jeans",     "Jeans",    1999,  880, "all",    "👖"),
    ("Black Skinny Jeans",         "Jeans",    1899,  820, "all",    "👖"),
    ("Formal Grey Trousers",       "Trousers", 1499,  650, "all",    "👔"),
    ("Beige Chinos",               "Trousers", 1399,  600, "all",    "👔"),
    ("Utility Cargo Trousers",     "Trousers", 1599,  700, "all",    "👔"),
    ("Cotton Jogger Lower",        "Lower",     799,  330, "all",    "🩳"),
    ("Soft Night Pyjama Lower",    "Lower",     599,  240, "all",    "🩳"),
    ("Active Summer Shorts",       "Lower",     649,  270, "summer", "🩳"),
    ("Fleece Winter Track Suit",   "Track Suit", 2499, 1100, "winter", "🧥"),
    ("Sports Dry-Fit Track Suit",  "Track Suit", 2199,  950, "all",    "🏃"),
    ("Thermal Zip Track Suit",     "Track Suit", 2799, 1200, "winter", "🧥"),
    ("Cotton Briefs (Pack of 3)",  "Underwear",  499,  200, "all",    "🩲"),
    ("Boxer Shorts (Pack of 3)",   "Underwear",  599,  250, "all",    "🩲"),
    ("Thermal Innerwear Set",      "Underwear",  899,  380, "winter", "🧦"),
]
BASE_DEMAND = {"T-Shirt": 40, "Jeans": 25, "Trousers": 20, "Lower": 30, "Track Suit": 12, "Underwear": 45}
ELASTICITY = {"T-Shirt": 0.030, "Jeans": 0.030, "Trousers": 0.028, "Lower": 0.032,
              "Track Suit": 0.045, "Underwear": 0.020}


def build_products():
    df = pd.DataFrame(_CATALOGUE, columns=["name", "category", "price", "cost", "season", "emoji"])
    df.insert(0, "product_id", range(1, len(df) + 1))
    return df


def load_products():
    DATA_DIR.mkdir(exist_ok=True)
    if not PRODUCTS_CSV.exists():
        build_products().to_csv(PRODUCTS_CSV, index=False)
    return pd.read_csv(PRODUCTS_CSV)


def _seasonality(season, month):
    if season == "winter":                      # peaks in Dec/Jan
        return 0.25 + max(0.0, np.cos(2 * np.pi * (month - 1) / 12))
    if season == "summer":                      # peaks in May
        return 0.25 + max(0.0, np.cos(2 * np.pi * (month - 5) / 12))
    return 1.0


def generate_sales(seed=42, samples_per_cell=6):
    """Synthetic monthly sales: demand = base * price effect * season * exp(elasticity * discount) * noise."""
    rng = np.random.default_rng(seed)
    products = load_products()
    rows = []
    for p in products.itertuples():
        for month in range(1, 13):
            cal_disc, _ = get_discount(month, p.category, p.season)
            for _ in range(samples_per_cell):
                disc = cal_disc if rng.random() < 0.5 else int(rng.choice(range(0, 65, 5)))
                mean = (BASE_DEMAND[p.category] * (700 / p.price) ** 0.3
                        * _seasonality(p.season, month)
                        * np.exp(ELASTICITY[p.category] * disc)
                        * (1.15 if cal_disc > 0 else 1.0)
                        * rng.lognormal(0, 0.15))
                rows.append((p.product_id, p.category, p.season, month, p.price, p.cost,
                             disc, int(rng.poisson(mean))))
    return pd.DataFrame(rows, columns=["product_id", "category", "season", "month",
                                       "price", "cost", "discount_pct", "units_sold"])


def load_sales():
    DATA_DIR.mkdir(exist_ok=True)
    if not SALES_CSV.exists():
        generate_sales().to_csv(SALES_CSV, index=False)
    return pd.read_csv(SALES_CSV)


def apply_seasonal_pricing(products, month):
    """Add discount_pct, offer, final_price columns for the chosen month."""
    out = products.copy()
    res = [get_discount(month, r.category, r.season) for r in out.itertuples()]
    out["discount_pct"] = [r[0] for r in res]
    out["offer"] = [r[1] for r in res]
    out["final_price"] = [apply_discount(p, d) for p, d in zip(out["price"], out["discount_pct"])]
    return out


def make_template():
    """Sample CSV a user can edit and upload for batch predictions."""
    return pd.DataFrame({
        "category": ["Jeans", "Track Suit", "T-Shirt"],
        "season": ["all", "winter", "summer"],
        "price": [1799, 2499, 699],
        "cost": [800, 1100, 300],
        "month": [3, 3, 4],
        "discount_pct": [10, 40, 20],
    })
