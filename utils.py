"""Helper functions: seasonal offer calendar, discount rules, formatting, validation."""
import calendar

MONTHS = [calendar.month_name[i] for i in range(1, 13)]

# target can be: "all", a season ("winter" / "summer") or a category name
OFFERS = [
    {"name": "Republic Day Sale",        "months": [1],      "target": "all",       "discount": 15},
    {"name": "Winter Clearance",         "months": [2, 3],   "target": "winter",    "discount": 40},
    {"name": "Holi Colour Fest",         "months": [3],      "target": "T-Shirt",   "discount": 20},
    {"name": "Summer Starter",           "months": [4, 5],   "target": "summer",    "discount": 15},
    {"name": "End of Summer Sale",       "months": [7, 8],   "target": "summer",    "discount": 35},
    {"name": "Independence Day Offer",   "months": [8],      "target": "all",       "discount": 20},
    {"name": "Festive Combo (Innerwear)", "months": [9, 10], "target": "Underwear", "discount": 25},
    {"name": "Diwali Dhamaka",           "months": [10, 11], "target": "all",       "discount": 30},
    {"name": "Winter Launch Deal",       "months": [11, 12], "target": "winter",    "discount": 10},
    {"name": "New Year Sale",            "months": [12],     "target": "all",       "discount": 25},
]

MAX_DISCOUNT = 60


def _matches(offer, category, season):
    return offer["target"] in ("all", category, season)


def get_discount(month, category, season):
    """Best (non-stacking) offer for a product in a month -> (discount %, offer name)."""
    best_pct, best_name = 0, ""
    for o in OFFERS:
        if month in o["months"] and _matches(o, category, season) and o["discount"] > best_pct:
            best_pct, best_name = o["discount"], o["name"]
    return min(best_pct, MAX_DISCOUNT), best_name


def active_offers(month):
    return [o for o in OFFERS if month in o["months"]]


def apply_discount(price, pct):
    return round(price * (1 - pct / 100))


def format_inr(x):
    return f"₹{x:,.0f}"


def validate_columns(df, required):
    """Return a list of missing column names (empty list = valid)."""
    return [c for c in required if c not in df.columns]
