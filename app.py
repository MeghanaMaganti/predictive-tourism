# app.py — Predictive Tourism (Streamlit + SQLite + AI/ML, 15+ viz, CSV upload, 5-model lab)
import os, sqlite3, math, random, io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="Predictive Tourism – Demand & Resource Optimization",
                   layout="wide", page_icon="🧭")

st.markdown("""
<style>
.stMetric {background: rgba(255,255,255,0.65); padding: 14px 16px; border-radius: 16px;}
.section-card {background: rgba(255,255,255,0.9); padding: 16px 18px; border-radius: 18px; box-shadow: 0 4px 18px rgba(0,0,0,0.07);}
</style>
""", unsafe_allow_html=True)

# ------------------------- DATA CREATION -------------------------
REQUIRED_COLS = [
    "date","destination","lat","lon","is_weekend","month","dow",
    "temp_c","rain","marketing_spend","social_sentiment","price_index",
    "bookings","avg_party_size","channel","lead_time_days",
    "rooms_available","guides_available","vehicles_available"
]
DB_PATH = "tourism.db"

def _make_synthetic() -> pd.DataFrame:
    np.random.seed(42)
    random.seed(42)
    start_date = datetime(2023, 1, 1)
    end_date   = datetime(2025, 9, 30)
    dates = pd.date_range(start_date, end_date, freq="D")

    destinations = [
        {"name":"Goa","lat":15.2993,"lon":74.1240},
        {"name":"Jaipur","lat":26.9124,"lon":75.7873},
        {"name":"Kerala","lat":10.8505,"lon":76.2711},
        {"name":"Manali","lat":32.2432,"lon":77.1892},
        {"name":"Andaman","lat":11.7401,"lon":92.6586},
        {"name":"Varanasi","lat":25.3176,"lon":82.9739},
        {"name":"Agra","lat":27.1767,"lon":78.0081},
        {"name":"Leh","lat":34.1526,"lon":77.5771}
    ]
    channels = ["App","Web","Agent","Kiosk"]
    guide_supply = {"Goa":250,"Jaipur":180,"Kerala":220,"Manali":140,"Andaman":120,"Varanasi":160,"Agra":150,"Leh":90}
    room_supply  = {"Goa":2000,"Jaipur":1500,"Kerala":1700,"Manali":1200,"Andaman":900,"Varanasi":1400,"Agra":1300,"Leh":600}
    transport_supply = {"Goa":800,"Jaipur":600,"Kerala":700,"Manali":450,"Andaman":300,"Varanasi":500,"Agra":550,"Leh":220}

    rows=[]
    for d in dates:
        month = d.month
        dow = d.weekday()
        is_weekend = 1 if dow>=5 else 0
        for dest in destinations:
            base = {"Goa":900,"Jaipur":700,"Kerala":800,"Manali":500,"Andaman":350,"Varanasi":650,"Agra":620,"Leh":250}[dest["name"]]
            season = 1 + 0.25*np.sin((month-1)/12*2*np.pi)
            festival_boost = 1.0
            if month in [10,11]: festival_boost += 0.25
            if month in [12,1]:  festival_boost += 0.2
            if dest["name"] in ["Leh","Manali"] and month in [6,7,8]: festival_boost += 0.35
            weekend_boost = 1.2 if is_weekend else 1.0

            temp_baseline = {"Goa":30,"Jaipur":28,"Kerala":29,"Manali":15,"Andaman":28,"Varanasi":27,"Agra":27,"Leh":10}[dest["name"]]
            temp_season = temp_baseline + 10*np.sin((month-3)/12*2*np.pi) + np.random.normal(0,1)
            rain_prob = 0.15 if month in [6,7,8,9] else 0.05
            rain = int(np.random.rand()<rain_prob)

            mkt_spend = np.random.gamma(shape=2, scale=200) * (1.2 if month in [4,5,10,11] else 1.0)
            sentiment = float(np.clip(np.random.normal(0.1 if not rain else -0.05, 0.5), -1, 1))

            base_price = {"Goa":70,"Jaipur":55,"Kerala":60,"Manali":65,"Andaman":80,"Varanasi":45,"Agra":50,"Leh":85}[dest["name"]]
            price_index = base_price * (1 + 0.15*season + (0.1 if festival_boost>1 else -0.05)) + np.random.normal(0,5)

            demand_mu = base * season * festival_boost * weekend_boost
            demand_mu *= (1 + 0.03*sentiment) * (1 + 0.0008*mkt_spend)
            demand_mu *= (0.92 if rain else 1.0)
            demand_mu *= (1 - 0.002*(price_index-base_price))
            demand = max(0, int(np.random.normal(demand_mu, demand_mu*0.15)))
            lead_time = max(1, int(np.random.gamma(2,6)))

            rows.append({
                "date": d.date(),
                "destination": dest["name"],
                "lat": dest["lat"], "lon": dest["lon"],
                "is_weekend": is_weekend, "month": month, "dow": dow,
                "temp_c": round(temp_season,1), "rain": rain,
                "marketing_spend": round(mkt_spend,2), "social_sentiment": round(sentiment,3),
                "price_index": round(price_index,2), "bookings": demand,
                "avg_party_size": int(np.clip(np.random.normal(2.6,0.7),1,5)),
                "channel": random.choice(channels), "lead_time_days": lead_time,
                "rooms_available": room_supply[dest["name"]],
                "guides_available": guide_supply[dest["name"]],
                "vehicles_available": transport_supply[dest["name"]],
            })
    return pd.DataFrame(rows)

def _create_table(cur):
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS tourism_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {", ".join([c+" TEXT" if c in ["date","destination","channel"] else c+" REAL" for c in REQUIRED_COLS]).replace(" REAL","")}
    );
    """)  # quick schema safeguard (we'll write via pandas)

def _save_df_to_sqlite(df: pd.DataFrame):
    con = sqlite3.connect(DB_PATH)
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_out.to_sql("tourism_daily", con, if_exists="replace", index=False)
    con.close()

def _regenerate_clean_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    _create_table(cur); con.commit()
    df = _make_synthetic()
    _save_df_to_sqlite(df)

def ensure_sqlite():
    if not os.path.exists(DB_PATH):
        _regenerate_clean_db()
        return
    try:
        con = sqlite3.connect(DB_PATH)
        pd.read_sql_query("SELECT COUNT(*) FROM tourism_daily", con)
        con.close()
    except Exception:
        _regenerate_clean_db()

@st.cache_data(show_spinner=False)
def load_df():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM tourism_daily", con)
    con.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df

def validate_and_prepare_upload(user_df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in user_df.columns}
    # normalize to REQUIRED_COLS if case differs
    missing = [c for c in REQUIRED_COLS if c not in [k for k in cols.keys()]]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # rename to exact required casing
    rename_map = {cols[c]: c for c in REQUIRED_COLS}
    df = user_df.rename(columns=rename_map).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # coerce numeric
    for c in REQUIRED_COLS:
        if c == "date" or c in ["destination","channel"]: continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bookings"])
    return df[REQUIRED_COLS].reset_index(drop=True)

ensure_sqlite()
df = load_df()

# ------------------------- SIDEBAR: CSV UPLOAD -------------------------
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload CSV to replace dataset", type=["csv"])
if up is not None:
    try:
        user_df = pd.read_csv(up)
        clean = validate_and_prepare_upload(user_df)
        _save_df_to_sqlite(clean)
        st.sidebar.success(f"Uploaded {len(clean):,} rows. Reloading…")
        st.cache_data.clear()
        df = load_df()
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

# ------------------------- SIDEBAR FILTERS -------------------------
st.sidebar.header("Filters")
dest_opts = ["All"] + sorted(df["destination"].unique().tolist())
dest = st.sidebar.selectbox("Destination", dest_opts, index=0)
_min_date, _max_date = df["date"].min(), df["date"].max()
if pd.isna(_min_date): _min_date = pd.Timestamp.today().normalize()
if pd.isna(_max_date): _max_date = pd.Timestamp.today().normalize()
start_date = st.sidebar.date_input("Start Date", value=_min_date.date())
end_date   = st.sidebar.date_input("End Date", value=_max_date.date())
channel = st.sidebar.multiselect("Channels", df["channel"].unique().tolist(), default=list(df["channel"].unique()))
show_map = st.sidebar.checkbox("Show map", value=True)

mask = df["date"].between(pd.to_datetime(start_date), pd.to_datetime(end_date)) & df["channel"].isin(channel)
if dest != "All":
    mask &= (df["destination"] == dest)
fdf = df.loc[mask].copy()

# ------------------------- KPIs -------------------------
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Bookings", f"{int(fdf['bookings'].sum()):,}")
c2.metric("Avg. Price Index", f"{fdf['price_index'].mean():.1f}")
c3.metric("Avg. Sentiment", f"{fdf['social_sentiment'].mean():.2f}")
c4.metric("Rainy Days %", f"{100*fdf['rain'].mean():.1f}%")

st.title("🧭 Predictive Tourism – Demand & Resource Optimization")
tabs = st.tabs(["Overview Dashboard","Forecasting (AI/ML)","Model Lab (5 Algos)","Resource Planner","Sustainability","Community"])

# helper: quick linear fit without statsmodels
def add_np_fit(fig, x, y, name="Linear fit"):
    if len(x) < 2: return fig
    try:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 100)
        ys = coeffs[0]*xs + coeffs[1]
        fig.add_trace(go.Scatter(x=xs, y=ys, name=name, mode="lines"))
    except Exception:
        pass
    return fig

# ------------------------- OVERVIEW (15+ viz) -------------------------
with tabs[0]:
    st.subheader("Key Trends & Insights")

    g1 = fdf.groupby("date", as_index=False)["bookings"].sum()
    fig1 = px.line(g1, x="date", y="bookings", title="Daily Bookings Trend"); st.plotly_chart(fig1, use_container_width=True)

    g1["roll30"] = g1["bookings"].rolling(30).mean()
    fig2 = px.line(g1, x="date", y="roll30", title="Rolling 30-Day Average Bookings"); st.plotly_chart(fig2, use_container_width=True)

    m = fdf.groupby(fdf["date"].dt.to_period("M"))["bookings"].sum().reset_index()
    m["date"] = m["date"].dt.to_timestamp()
    fig3 = px.bar(m, x="date", y="bookings", title="Monthly Bookings"); st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(fdf, x=fdf["date"].dt.day_name(), y="bookings", title="Bookings Distribution by Weekday"); st.plotly_chart(fig4, use_container_width=True)

    pie = fdf.groupby("destination")["bookings"].sum().reset_index()
    fig5 = px.pie(pie, names="destination", values="bookings", title="Destination Share of Bookings"); st.plotly_chart(fig5, use_container_width=True)

    ch = fdf.groupby("channel")["bookings"].sum().reset_index()
    fig6 = px.bar(ch, x="channel", y="bookings", title="Bookings by Channel"); st.plotly_chart(fig6, use_container_width=True)

    fig7 = px.histogram(fdf, x="lead_time_days", nbins=30, title="Lead Time (days) Distribution"); st.plotly_chart(fig7, use_container_width=True)

    agg = fdf.groupby("date")[["bookings","price_index","marketing_spend","temp_c","social_sentiment"]].mean().reset_index()
    fig8 = px.scatter(agg, x="price_index", y="bookings", title="Price vs Bookings")
    fig8 = add_np_fit(fig8, agg["price_index"].values, agg["bookings"].values)
    st.plotly_chart(fig8, use_container_width=True)

    fig9 = px.scatter(agg, x="social_sentiment", y="bookings", title="Sentiment vs Bookings")
    fig9 = add_np_fit(fig9, agg["social_sentiment"].values, agg["bookings"].values)
    st.plotly_chart(fig9, use_container_width=True)

    fig10 = px.scatter(agg, x="temp_c", y="bookings", title="Temperature vs Bookings")
    fig10 = add_np_fit(fig10, agg["temp_c"].values, agg["bookings"].values)
    st.plotly_chart(fig10, use_container_width=True)

    corr = fdf[["bookings","price_index","marketing_spend","temp_c","lead_time_days","is_weekend","rain"]].corr()
    fig11 = px.imshow(corr, text_auto=True, title="Feature Correlation"); st.plotly_chart(fig11, use_container_width=True)

    fig12 = px.bar(fdf.groupby("destination")["avg_party_size"].mean().reset_index(),
                   x="destination", y="avg_party_size", title="Avg Party Size by Destination"); st.plotly_chart(fig12, use_container_width=True)

    if show_map:
        loc = fdf.groupby(["destination","lat","lon"])["bookings"].mean().reset_index()
        fig13 = px.scatter_mapbox(loc, lat="lat", lon="lon", size="bookings", color="destination",
                                  zoom=3, height=450, title="Destinations Map (avg bookings)", hover_name="destination")
        fig13.update_layout(mapbox_style="open-street-map"); st.plotly_chart(fig13, use_container_width=True)

    rain_grp = fdf.groupby("rain")["bookings"].mean().reset_index()
    rain_grp["rain"] = rain_grp["rain"].map({0:"No Rain",1:"Rain"})
    fig14 = px.bar(rain_grp, x="rain", y="bookings", title="Rain Impact on Avg Bookings"); st.plotly_chart(fig14, use_container_width=True)

    util = fdf.groupby("destination").agg(
        demand=("bookings","mean"),
        rooms=("rooms_available","mean"),
        guides=("guides_available","mean"),
        vehicles=("vehicles_available","mean")
    ).reset_index()
    util["room_util_%"] = 100 * util["demand"] / util["rooms"]
    fig15 = px.bar(util, x="destination", y="room_util_%", title="Avg Room Utilization (%) by Destination"); st.plotly_chart(fig15, use_container_width=True)

# ------------------------- FORECASTING (simple RF like before) -------------------------
with tabs[1]:
    st.subheader("Train & Forecast (Random Forest)")
    f_dest = st.selectbox("Destination for modeling", sorted(df["destination"].unique()))
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

    sdf = df[df["destination"]==f_dest].sort_values("date").copy()
    sdf["dayofyear"] = sdf["date"].dt.dayofyear.fillna(1).astype(int)
    sdf["weekofyear"] = sdf["date"].dt.isocalendar().week.astype(int)
    sdf["year"] = sdf["date"].dt.year
    for lag in [1,7,14]:
        sdf[f"lag_{lag}"] = sdf["bookings"].shift(lag)
    sdf["roll7"] = sdf["bookings"].rolling(7).mean()
    sdf["roll14"] = sdf["bookings"].rolling(14).mean()
    sdf = sdf.dropna().reset_index(drop=True)

    features = ["is_weekend","month","dow","temp_c","rain","marketing_spend","social_sentiment","price_index",
                "lead_time_days","dayofyear","weekofyear","year","lag_1","lag_7","lag_14","roll7","roll14"]
    if sdf.empty:
        st.warning("Not enough data after feature engineering.")
    else:
        X = sdf[features]; y = sdf["bookings"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
        preds = rf.predict(X_test)
        cA,cB = st.columns(2)
        cA.metric("MAE", f"{mean_absolute_error(y_test, preds):.1f}")
        cB.metric("R²", f"{r2_score(y_test, preds):.3f}")

        back = pd.DataFrame({"date": sdf.iloc[X_test.index]["date"].values, "actual": y_test.values, "RF": preds})
        fig_back = go.Figure()
        fig_back.add_trace(go.Scatter(x=back["date"], y=back["actual"], name="Actual"))
        fig_back.add_trace(go.Scatter(x=back["date"], y=back["RF"], name="Predicted (RF)"))
        fig_back.update_layout(title=f"Backtest – {f_dest}")
        st.plotly_chart(fig_back, use_container_width=True)

        # Forecast
        last_known = sdf.iloc[-1].copy()
        future_dates = pd.date_range(sdf["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        fut = []
        lag1, lag7, lag14 = last_known["bookings"], sdf.iloc[-7]["bookings"], sdf.iloc[-14]["bookings"]
        roll7, roll14 = sdf["bookings"].tail(7).mean(), sdf["bookings"].tail(14).mean()
        for d_ in future_dates:
            row = {
                "date": d_,
                "is_weekend": 1 if d_.weekday()>=5 else 0,
                "month": int(d_.month),
                "dow": int(d_.weekday()),
                "temp_c": float(np.clip(last_known["temp_c"] + np.random.normal(0,1), -10, 45)),
                "rain": int(np.random.rand() < (0.15 if d_.month in [6,7,8,9] else 0.05)),
                "marketing_spend": float(max(0, last_known["marketing_spend"] * np.random.uniform(0.9,1.1))),
                "social_sentiment": float(np.clip(last_known["social_sentiment"] + np.random.normal(0,0.1), -1, 1)),
                "price_index": float(max(1, last_known["price_index"] + np.random.normal(0,2))),
                "lead_time_days": int(max(1, last_known["lead_time_days"] + np.random.randint(-2,3))),
                "dayofyear": int(d_.dayofyear),
                "weekofyear": int(d_.isocalendar().week),
                "year": int(d_.year),
                "lag_1": float(lag1), "lag_7": float(lag7), "lag_14": float(lag14),
                "roll7": float(roll7), "roll14": float(roll14)
            }
            Xf = pd.DataFrame([row])[features]
            yhat = float(rf.predict(Xf)[0])
            row["pred"] = max(0, yhat)
            lag14, lag7, lag1 = lag7, lag1, row["pred"]
            roll7 = (roll7*7 - sdf["bookings"].iloc[-7] + row["pred"]) / 7 if len(sdf)>=7 else row["pred"]
            roll14 = (roll14*14 - sdf["bookings"].iloc[-14] + row["pred"]) / 14 if len(sdf)>=14 else row["pred"]
            fut.append(row)
        fut = pd.DataFrame(fut)
        fig_fore = px.line(fut, x="date", y="pred", title=f"{f_dest} – {horizon}-Day Forecast (RF)")
        st.plotly_chart(fig_fore, use_container_width=True)

# ------------------------- MODEL LAB (5 algorithms + comparison) -------------------------
with tabs[2]:
    st.subheader("Model Lab – Train 5 Algorithms & Compare")
    lab_dest = st.selectbox("Destination for Model Lab", sorted(df["destination"].unique()), key="lab_dest")
    test_size = st.slider("Test size (%)", 10, 40, 20, 1)

    lab = df[df["destination"]==lab_dest].sort_values("date").copy()
    # features
    lab["dayofyear"] = lab["date"].dt.dayofyear.fillna(1).astype(int)
    lab["weekofyear"] = lab["date"].dt.isocalendar().week.astype(int)
    lab["year"] = lab["date"].dt.year
    for lag in [1,7,14]:
        lab[f"lag_{lag}"] = lab["bookings"].shift(lag)
    lab["roll7"] = lab["bookings"].rolling(7).mean()
    lab["roll14"] = lab["bookings"].rolling(14).mean()
    lab = lab.dropna().reset_index(drop=True)

    features = ["is_weekend","month","dow","temp_c","rain","marketing_spend","social_sentiment","price_index",
                "lead_time_days","dayofyear","weekofyear","year","lag_1","lag_7","lag_14","roll7","roll14"]
    if lab.empty:
        st.warning("Not enough rows for training. Change destination or filters.")
    else:
        X = lab[features]; y = lab["bookings"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, shuffle=False)

        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=250, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "KNN": KNeighborsRegressor(n_neighbors=5),
        }

        results = []
        preds_store = {}
        for name, mdl in models.items():
            mdl.fit(X_train, y_train)
            p = mdl.predict(X_test)
            preds_store[name] = p
            results.append({"Model": name, "MAE": mean_absolute_error(y_test, p), "R2": r2_score(y_test, p)})
        res_df = pd.DataFrame(results).sort_values("MAE")

        cA,cB = st.columns(2)
        cA.dataframe(res_df, use_container_width=True)
        fig_cmp = px.bar(res_df, x="Model", y="MAE", title="MAE by Model (lower is better)")
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Parity plot for best model
        best = res_df.iloc[0]["Model"]
        parity = pd.DataFrame({"Actual": y_test.values, "Pred": preds_store[best]})
        fig_par = px.scatter(parity, x="Actual", y="Pred", title=f"Parity Plot – {best}")
        fig_par = add_np_fit(fig_par, parity["Actual"].values, parity["Pred"].values, name="y=x fit")
        st.plotly_chart(fig_par, use_container_width=True)

        # Residuals plot for best model
        residuals = parity["Actual"] - parity["Pred"]
        fig_res = px.histogram(residuals, nbins=40, title=f"Residuals – {best}")
        st.plotly_chart(fig_res, use_container_width=True)

        # Feature importance (tree models)
        if best in ["RandomForest","GradientBoosting"]:
            imp = models[best].feature_importances_
            imp_df = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance", ascending=False)
            fig_imp = px.bar(imp_df, x="feature", y="importance", title=f"Feature Importance – {best}")
            st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------- RESOURCE PLANNER -------------------------
with tabs[3]:
    st.subheader("Capacity & Staffing Recommendations")
    r_dest = st.selectbox("Destination for planning", sorted(df["destination"].unique()), key="plan_dest")
    lookback_days = st.slider("Lookback window (days)", 7, 90, 30, 1)
    planning_df = df[(df["destination"]==r_dest) & (df["date"]>=df["date"].max()-pd.Timedelta(days=lookback_days))]
    demand_avg = planning_df["bookings"].mean()

    rooms = int(planning_df["rooms_available"].iloc[-1])
    guides = int(planning_df["guides_available"].iloc[-1])
    vehicles = int(planning_df["vehicles_available"].iloc[-1])

    guide_capacity = st.number_input("Tourists per guide per day", 6, 50, 15)
    vehicle_capacity = st.number_input("Tourists per vehicle per trip", 2, 60, 20)
    occupancy_target = st.slider("Target room occupancy (%)", 50, 98, 85)

    needed_rooms = math.ceil(demand_avg / (occupancy_target/100))
    needed_guides = math.ceil(demand_avg / guide_capacity)
    needed_vehicles = math.ceil(demand_avg / vehicle_capacity)

    c1,c2,c3 = st.columns(3)
    c1.metric("Avg Demand (last window)", f"{int(demand_avg)}")
    c2.metric("Rooms Needed", f"{needed_rooms} (cap {rooms})")
    c3.metric("Guides Needed", f"{needed_guides} (cap {guides})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Vehicles Needed", f"{needed_vehicles} (cap {vehicles})")
    c2.metric("Room Gap", f"{max(0, needed_rooms - rooms)}")
    c3.metric("Guide Gap", f"{max(0, needed_guides - guides)}")

    bar = pd.DataFrame({
        "Resource":["Rooms","Guides","Vehicles"],
        "Capacity":[rooms,guides,vehicles],
        "Required":[needed_rooms,needed_guides,needed_vehicles]
    })
    fig_cap = go.Figure(data=[
        go.Bar(name="Capacity", x=bar["Resource"], y=bar["Capacity"]),
        go.Bar(name="Required", x=bar["Resource"], y=bar["Required"])
    ])
    fig_cap.update_layout(barmode="group", title=f"{r_dest} – Capacity vs Required")
    st.plotly_chart(fig_cap, use_container_width=True)

# ------------------------- SUSTAINABILITY -------------------------
with tabs[4]:
    st.subheader("Sustainability Metrics & Insights")
    sdf = fdf.copy()
    sdf["emissions_kg_co2e"] = sdf["bookings"] * (6.0 + 0.5*sdf["rain"])
    denom = sdf["bookings"].replace(0, np.nan)
    sdf["eco_score"] = np.clip(100 - (sdf["emissions_kg_co2e"]/denom).fillna(0)*2, 0, 100)

    g = sdf.groupby("date", as_index=False)[["emissions_kg_co2e","bookings"]].sum()
    g["co2_per_booking"] = g["emissions_kg_co2e"] / g["bookings"].replace(0, np.nan)

    fig_s1 = px.line(g, x="date", y="emissions_kg_co2e", title="Daily Estimated Emissions (kg CO₂e)")
    fig_s2 = px.line(g, x="date", y="co2_per_booking", title="CO₂e per Booking")
    st.plotly_chart(fig_s1, use_container_width=True); st.plotly_chart(fig_s2, use_container_width=True)

    dest_eco = sdf.groupby("destination")[["eco_score"]].mean().reset_index()
    fig_s3 = px.bar(dest_eco, x="destination", y="eco_score", title="Eco Score by Destination (↑ better)")
    st.plotly_chart(fig_s3, use_container_width=True)
    st.info("Promote off-peak travel, pooled transport, and eco-stays to reduce emissions.")

# ------------------------- COMMUNITY -------------------------
with tabs[5]:
    st.subheader("Community Engagement (Demo)")
    st.write("Enable local communities to participate and benefit from tourism planning.")
    idea = st.text_area("Share a program idea (training, crafts, local tours, eco-drives):")
    if st.button("Submit idea"):
        st.success("Thanks! Your idea has been noted for review by the tourism board.")
    dest_rev = fdf.groupby("destination")["bookings"].sum().reset_index()
    dest_rev["community_share_est"] = dest_rev["bookings"] * 12.5  # demo ₹ per booking
    fig_c = px.bar(dest_rev, x="destination", y="community_share_est",
                   title="Estimated Community Revenue Share (demo)", labels={"community_share_est":"₹"})
    st.plotly_chart(fig_c, use_container_width=True)
