"""Clothics – Smart Fashion Store (Streamlit). Run: streamlit run app.py"""
import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import (load_products, load_sales, apply_seasonal_pricing, make_template)
from model import train_model, predict_batch, optimize_discount
from utils import MONTHS, OFFERS, active_offers, get_discount, format_inr, validate_columns

st.set_page_config(page_title="Clothics – Smart Fashion Store", page_icon="👕", layout="wide")
st.markdown("<style>.big-emoji{font-size:64px;text-align:center}</style>", unsafe_allow_html=True)


@st.cache_data
def get_products():
    return load_products()


@st.cache_resource
def get_model():
    return train_model(load_sales())


products = get_products()
model, metrics = get_model()
if "cart" not in st.session_state:
    st.session_state.cart = {}


def add_to_cart(pid):
    st.session_state.cart[pid] = st.session_state.cart.get(pid, 0) + 1


def clear_cart():
    st.session_state.cart = {}


# ---------------- Sidebar ----------------
st.sidebar.title("👕 Clothics")
month = st.sidebar.selectbox("Shopping month (simulate season)", range(1, 13),
                             index=dt.date.today().month - 1, format_func=lambda m: MONTHS[m - 1])
catalog = apply_seasonal_pricing(products, month)

st.sidebar.markdown("### 🛒 Cart")
total = 0
for pid, qty in st.session_state.cart.items():
    row = catalog[catalog.product_id == pid].iloc[0]
    st.sidebar.write(f"{row.emoji} {row['name']} × {qty} — {format_inr(row.final_price * qty)}")
    total += row.final_price * qty
if st.session_state.cart:
    st.sidebar.markdown(f"**Total: {format_inr(total)}**")
    if st.sidebar.button("Checkout ✅"):
        st.balloons()
        st.sidebar.success("Order placed (demo)!")
    st.sidebar.button("Clear cart", on_click=clear_cart)
else:
    st.sidebar.caption("Your cart is empty.")

# ---------------- Header ----------------
st.title("Clothics – Smart Seasonal Fashion Store")
for o in active_offers(month):
    st.success(f"🎉 **{o['name']}** – {o['discount']}% off on {o['target']} items ({MONTHS[month-1]})")
if not active_offers(month):
    st.info("No special offer this month – explore the Pricing Lab to see what discount would work best.")

tab_shop, tab_cal, tab_lab, tab_batch = st.tabs(
    ["🛍️ Shop", "📅 Offer Calendar", "🤖 AI Pricing Lab", "📂 Batch Predict (CSV)"])

# ---------------- Shop ----------------
with tab_shop:
    c1, c2 = st.columns([2, 1])
    cats = c1.multiselect("Category", sorted(catalog.category.unique()), default=sorted(catalog.category.unique()))
    sort = c2.selectbox("Sort by", ["Biggest discount", "Price: low to high", "Price: high to low"])
    view = catalog[catalog.category.isin(cats)]
    view = {"Biggest discount": view.sort_values("discount_pct", ascending=False),
            "Price: low to high": view.sort_values("final_price"),
            "Price: high to low": view.sort_values("final_price", ascending=False)}[sort]
    cols = st.columns(4)
    for i, row in enumerate(view.itertuples()):
        with cols[i % 4]:
            with st.container(border=True):
                st.markdown(f"<div class='big-emoji'>{row.emoji}</div>", unsafe_allow_html=True)
                st.markdown(f"**{row.name}**")
                st.caption(f"{row.category} • {row.season} wear")
                if row.discount_pct > 0:
                    st.markdown(f"~~{format_inr(row.price)}~~ **{format_inr(row.final_price)}** "
                                f":red[{row.discount_pct}% OFF]")
                    st.caption(f"🎉 {row.offer}")
                else:
                    st.markdown(f"**{format_inr(row.price)}**")
                    st.caption("Regular price")
                st.button("Add to cart", key=f"add_{row.product_id}", on_click=add_to_cart, args=(row.product_id,))

# ---------------- Offer calendar ----------------
with tab_cal:
    st.subheader("Seasonal & festival discount rules")
    st.dataframe(pd.DataFrame([{"Offer": o["name"], "Months": ", ".join(MONTHS[m - 1][:3] for m in o["months"]),
                                "Applies to": o["target"], "Discount %": o["discount"]} for o in OFFERS]),
                 hide_index=True)
    heat = pd.DataFrame({m: {c: get_discount(m, c, "winter" if c == "Track Suit" else "all")[0]
                              for c in sorted(products.category.unique())} for m in range(1, 13)})
    heat.columns = [MONTHS[m - 1][:3] for m in range(1, 13)]
    st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Reds",
                              title="Discount % by category and month (winter wear for Track Suit)"))

# ---------------- AI Pricing Lab ----------------
with tab_lab:
    st.subheader("Demand forecast & profit-maximising discount")
    st.caption(f"Model: Gradient Boosting • trained on {metrics['rows']} sales rows • "
               f"MAE {metrics['MAE']:.1f} units • R² {metrics['R2']:.2f}")
    a, b, c = st.columns(3)
    name = a.selectbox("Product", products["name"])
    lab_month = b.selectbox("Month", range(1, 13), index=month - 1, format_func=lambda m: MONTHS[m - 1], key="lab_m")
    max_d = c.slider("Max discount to test (%)", 10, 60, 60, step=5)
    prod = products[products["name"] == name].iloc[0]
    curve, best = optimize_discount(model, prod, lab_month, max_d)
    rule_d, rule_name = get_discount(lab_month, prod.category, prod.season)

    m1, m2, m3 = st.columns(3)
    m1.metric("Best discount (profit)", f"{int(best.discount_pct)}%")
    m2.metric("Expected units / month", f"{best.pred_units:.0f}")
    m3.metric("Expected profit", format_inr(best.pred_profit))
    st.caption(f"Calendar rule today: **{rule_d}%** {('(' + rule_name + ')') if rule_name else '(no offer)'}")

    l, r = st.columns(2)
    l.plotly_chart(px.line(curve, x="discount_pct", y="pred_units", markers=True,
                           title="Predicted units vs discount"))
    r.plotly_chart(px.line(curve, x="discount_pct", y=["pred_revenue", "pred_profit"], markers=True,
                           title="Revenue & profit vs discount"))

    fc = pd.DataFrame({"Month": [MONTHS[m - 1][:3] for m in range(1, 13)]})
    rows = [{"category": prod.category, "season": prod.season, "price": prod.price, "cost": prod.cost,
             "month": m, "discount_pct": get_discount(m, prod.category, prod.season)[0]} for m in range(1, 13)]
    fc["Predicted units"] = predict_batch(model, pd.DataFrame(rows))["pred_units"]
    st.plotly_chart(px.bar(fc, x="Month", y="Predicted units",
                           title=f"12-month demand forecast with calendar offers – {name}"))

# ---------------- Batch predict ----------------
with tab_batch:
    st.subheader("Upload a CSV and get live demand predictions")
    required = ["category", "season", "price", "cost", "month", "discount_pct"]
    st.write("Required columns:", ", ".join(required))
    st.download_button("Download template CSV", make_template().to_csv(index=False), "template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        df = pd.read_csv(up)
        missing = validate_columns(df, required)
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            res = predict_batch(model, df)
            st.dataframe(res)
            st.plotly_chart(px.bar(res, x=res.index.astype(str), y="pred_units", color="category",
                                   title="Predicted units per row"))
            st.download_button("Download predictions", res.to_csv(index=False), "predictions.csv", "text/csv")
