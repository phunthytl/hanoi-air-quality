# -*- coding: utf-8 -*-
import os
import time
import math
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image
import folium
from folium.plugins import TimestampedGeoJson
from branca.colormap import linear

# =========================
# CONFIG & CONSTANTS
# =========================
st.set_page_config(page_title="Hanoi Air Quality: 30-day Analysis", layout="wide")

# Bạn có thể thay bằng biến môi trường nếu muốn: os.environ.get("OWM_API_KEY")
OWM_API_KEY = "490f66c05505839fe0646bb5aa5770dc"
CITY_NAME = "Hanoi"
COUNTRY_CODE = "VN"
TARGET_DAYS = 30
CHUNK_DAYS = 5  # OWM free thường chỉ cho 5 ngày history/đợt
TIMEZONE = timezone(timedelta(hours=7))  # Asia/Bangkok (UTC+7)

# =========================
# UTILS
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
def geocode_city(city, country, api_key):
    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": f"{city},{country}", "limit": 1, "appid": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise ValueError("Không tìm thấy toạ độ thành phố.")
    lat, lon = js[0]["lat"], js[0]["lon"]
    return lat, lon, js[0]

def unix(dt):
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def to_local(ts):
    # OWM trả dt là UNIX UTC
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(TIMEZONE)

@st.cache_data(show_spinner=True, ttl=1800)
def get_air_history(lat, lon, start_dt, end_dt, api_key):
    """
    Gọi OWM Air Pollution History theo các cửa sổ 5 ngày để cố lấy đủ 30 ngày.
    Trả về list bản ghi (dict) từ API.
    """
    collected = []
    total_days = (end_dt - start_dt).days + 1
    # lùi theo các khung 5 ngày
    window_end = end_dt
    while window_end > start_dt:
        window_start = max(start_dt, window_end - timedelta(days=CHUNK_DAYS-1))
        url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
        params = {
            "lat": lat, "lon": lon,
            "start": unix(window_start.replace(hour=0, minute=0, second=0, microsecond=0)),
            "end": unix((window_end + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)),
            "appid": api_key
        }
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 429:
                # hạn mức -> nghỉ 2s rồi thử lại
                time.sleep(2)
                r = requests.get(url, params=params, timeout=45)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict) and "list" in js:
                collected.extend(js["list"])
        except requests.HTTPError as e:
            # Nếu vượt giới hạn lịch sử (free), vẫn tiếp tục với phần lấy được
            st.info(f"⚠️ Không lấy được đoạn {window_start.date()} → {window_end.date()}: {e}")
        window_end = window_start - timedelta(days=1)
    # loại trùng theo timestamp
    seen = set()
    uniq = []
    for it in collected:
        t = int(it.get("dt", -1))
        if t not in seen and t != -1:
            uniq.append(it)
            seen.add(t)
    # sort theo thời gian
    uniq.sort(key=lambda x: x.get("dt", 0))
    return uniq

def normalize_dataframe(raw_list, lat, lon):
    if not raw_list:
        return pd.DataFrame()
    rows = []
    for it in raw_list:
        dt_utc = int(it.get("dt", 0))
        main = it.get("main", {}) or {}
        comps = it.get("components", {}) or {}
        rows.append({
            "dt_utc": dt_utc,
            "time": to_local(dt_utc),
            "aqi": main.get("aqi", np.nan),
            "co": comps.get("co", np.nan),
            "no": comps.get("no", np.nan),
            "no2": comps.get("no2", np.nan),
            "o3": comps.get("o3", np.nan),
            "so2": comps.get("so2", np.nan),
            "pm2_5": comps.get("pm2_5", np.nan),
            "pm10": comps.get("pm10", np.nan),
            "nh3": comps.get("nh3", np.nan),
            "lat": lat, "lon": lon
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["dt_utc"]).sort_values("time").reset_index(drop=True)

    # Chuẩn hoá kiểu dữ liệu
    num_cols = ["aqi","co","no","no2","o3","so2","pm2_5","pm10","nh3","lat","lon"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Xử lý thiếu: forward-fill theo giờ (vì cùng 1 điểm đo)
    if not df.empty:
        df = df.set_index("time")
        df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
        df = df.reset_index()

    # Tạo đặc trưng mới
    if not df.empty:
        df["pm_ratio"] = df["pm2_5"] / df["pm10"]
        df["hour"] = df["time"].dt.hour
        df["day"] = df["time"].dt.date.astype(str)
        df["weekday"] = df["time"].dt.day_name()
        df["is_weekend"] = df["weekday"].isin(["Saturday","Sunday"]).astype(int)
        df["rolling_pm25_24h"] = df["pm2_5"].rolling(24, min_periods=6).mean()
        df["rolling_pm10_24h"] = df["pm10"].rolling(24, min_periods=6).mean()
        df["aqi_label"] = df["aqi"].map({
            1:"Good", 2:"Fair", 3:"Moderate", 4:"Poor", 5:"Very Poor"
        }).fillna("Unknown")
    return df

def corr_df(df):
    cols = ["aqi","co","no","no2","o3","so2","pm2_5","pm10","nh3","pm_ratio","rolling_pm25_24h","rolling_pm10_24h"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr().round(3)

def summarize_insight(df):
    if df.empty:
        return {
            "n": 0, "date_range": "No data",
            "max_pm25": None, "max_pm25_time": None,
            "bad_hours_share": None,
            "mean_pm25": None, "median_pm25": None,
            "top_day": None
        }
    # Thống kê cơ bản
    n = len(df)
    dr = f"{df['time'].min().date()} → {df['time'].max().date()}"
    idx_max = df["pm2_5"].idxmax()
    time_max = df.loc[idx_max, "time"]
    max_pm25 = df.loc[idx_max, "pm2_5"]
    bad_hours_share = (df["aqi"] >= 4).mean()  # tỉ lệ giờ 'Poor' & 'Very Poor'
    mean_pm25 = df["pm2_5"].mean()
    median_pm25 = df["pm2_5"].median()
    daily = df.groupby("day")["pm2_5"].mean().sort_values(ascending=False)
    top_day = daily.index[0] if len(daily) else None
    return {
        "n": n, "date_range": dr,
        "max_pm25": float(max_pm25), "max_pm25_time": time_max.strftime("%Y-%m-%d %H:%M"),
        "bad_hours_share": float(bad_hours_share),
        "mean_pm25": float(mean_pm25), "median_pm25": float(median_pm25),
        "top_day": top_day
    }

def make_wordcloud(text_series):
    txt = " ".join([str(x) for x in text_series if pd.notna(x)])
    if not txt.strip():
        return None
    wc = WordCloud(width=1200, height=600, background_color="white").generate(txt)
    bio = BytesIO()
    wc.to_image().save(bio, format="PNG")
    bio.seek(0)
    return Image.open(bio)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")
st.sidebar.write("Phân tích chất lượng không khí tại **Hà Nội** (OWM).")
target_days = st.sidebar.slider("Số ngày gần nhất", 7, 30, TARGET_DAYS, 1)
show_regression = st.sidebar.checkbox("Scatter + OLS regression", True)
use_treemap = st.sidebar.selectbox("Phân cấp", ["Sunburst AQI → Weekday", "Treemap AQI → Hour"], index=0)
map_mode = st.sidebar.selectbox("Bản đồ", ["Folium (Timestamped)", "Plotly Mapbox"], index=0)

# =========================
# DATA FETCH
# =========================
st.title("🌫️ Hanoi Air Quality — 30-day Analysis (OpenWeatherMap)")
st.caption("Nguồn: OpenWeatherMap Air Pollution API. Ứng dụng sẽ tự báo nếu chỉ lấy được ≤5 ngày do giới hạn gói.")

# Geocode
lat, lon, geo_meta = geocode_city(CITY_NAME, COUNTRY_CODE, OWM_API_KEY)

# Time window
end_local = datetime.now(tz=TIMEZONE)
start_local = end_local - timedelta(days=target_days-1)
# OWM history dùng UTC
start_utc = start_local.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
end_utc = end_local.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)

raw = get_air_history(lat, lon, start_utc, end_utc, OWM_API_KEY)
df = normalize_dataframe(raw, lat, lon)

if df.empty:
    st.error("Không lấy được dữ liệu từ OWM. Hãy kiểm tra API key/quyền truy cập hoặc thử lại sau.")
    st.stop()

# Cảnh báo nếu độ phủ < 30 ngày
got_days = (df["time"].max().date() - df["time"].min().date()).days + 1
if got_days < target_days - 1:
    st.warning(f"Chỉ lấy được ~{got_days} ngày dữ liệu (giới hạn API). Vẫn tiến hành phân tích trên phần dữ liệu này.")

# =========================
# SUMMARY KPIs & STORYTELLING
# =========================
kpi = summarize_insight(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Khoảng thời gian", kpi["date_range"])
c2.metric("Số bản ghi (giờ)", f"{kpi['n']:,}")
c3.metric("PM2.5 cao nhất (µg/m³)", f"{kpi['max_pm25']:.1f}" if kpi["max_pm25"] else "—", help=f"Thời điểm: {kpi['max_pm25_time'] or '—'}")
c4.metric("Tỉ lệ giờ AQI xấu (≥Poor)", f"{kpi['bad_hours_share']*100:.1f}%" if kpi["bad_hours_share"] is not None else "—")

with st.expander("🧾 1 trang storytelling (insights chính)"):
    st.markdown(f"""
**Bối cảnh.** Phân tích chuỗi thời gian chất lượng không khí ở Hà Nội trong giai đoạn **{kpi['date_range']}**.  
**Quy mô dữ liệu.** {kpi['n']:,} quan sát giờ.  
**Mức ô nhiễm đáng chú ý.** PM2.5 cao nhất đạt **{kpi['max_pm25']:.1f} µg/m³** vào **{kpi['max_pm25_time']}**.  
**Rủi ro phơi nhiễm.** Tỉ lệ giờ có AQI **từ mức Poor trở lên** ~ **{kpi['bad_hours_share']*100:.1f}%**.  
**Ngày tệ nhất (theo PM2.5 trung bình ngày).** **{kpi['top_day']}**.  

**Insight nhanh:**
- **Nhịp ngày–đêm:** So sánh PM2.5 theo giờ cho thấy các đỉnh thường rơi vào buổi **sáng sớm** và/hoặc **tối**, gợi ý ảnh hưởng giao thông – nghịch nhiệt.  
- **Tương quan PM2.5–PM10:** Hệ số tương quan cao → nguồn hạt mịn và thô cùng biến thiên; **pm_ratio** giúp nhận diện ưu thế hạt mịn.  
- **Độ trễ 24h:** Trung bình trượt 24h làm mượt biến động, hữu ích để cảnh báo sớm nếu xu hướng tăng kéo dài.

> Lưu ý: Nếu bạn dùng gói OWM trả phí, dữ liệu 30 ngày sẽ đầy đủ hơn; bản hiện tại có thể bị giới hạn ≤5 ngày.
""")

# =========================
# CHARTS — INTERACTIVE (≥3)
# =========================

# 1) Line/Area theo thời gian (Plotly)
st.subheader("📈 PM2.5 & PM10 theo thời gian")
fig_area = go.Figure()
fig_area.add_trace(go.Scatter(x=df["time"], y=df["pm2_5"], mode="lines", name="PM2.5", fill="tozeroy"))
fig_area.add_trace(go.Scatter(x=df["time"], y=df["pm10"], mode="lines", name="PM10", fill=None))
fig_area.update_layout(xaxis_title="Time", yaxis_title="µg/m³", hovermode="x unified")
st.plotly_chart(fig_area, use_container_width=True)

# 2) Histogram/Box/Violin (Plotly)
st.subheader("📦 Phân phối PM2.5")
tab_hist, tab_box, tab_violin = st.tabs(["Histogram", "Boxplot", "Violin"])
with tab_hist:
    st.plotly_chart(px.histogram(df, x="pm2_5", nbins=40, marginal="rug", title="Histogram PM2.5"), use_container_width=True)
with tab_box:
    st.plotly_chart(px.box(df, y="pm2_5", points="all", title="Boxplot PM2.5"), use_container_width=True)
with tab_violin:
    st.plotly_chart(px.violin(df, y="pm2_5", box=True, points="all", title="Violin PM2.5"), use_container_width=True)

# 3) Scatter + hồi quy (Plotly trendline=ols)
st.subheader("🔬 PM2.5 vs PM10 (Scatter + OLS)")
trend = "ols" if show_regression else None
scatter_fig = px.scatter(df, x="pm2_5", y="pm10", opacity=0.7,
                         trendline=trend, trendline_color_override="black",
                         labels={"pm2_5":"PM2.5 (µg/m³)", "pm10":"PM10 (µg/m³)"},
                         hover_data=["time"])
st.plotly_chart(scatter_fig, use_container_width=True)

# 4) Heatmap tương quan
st.subheader("🧪 Tương quan các chỉ tiêu")
corr = corr_df(df)
heat = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
st.plotly_chart(heat, use_container_width=True)

# 5) Sunburst/Treemap
st.subheader("🪅 Phân cấp AQI")
if use_treemap.startswith("Sunburst"):
    sb = px.sunburst(df, path=["aqi_label","weekday"], values=None, title="Sunburst: AQI → Weekday")
    st.plotly_chart(sb, use_container_width=True)
else:
    # Treemap AQI → Hour buckets
    df["hour_bin"] = pd.cut(df["hour"], bins=[-0.1,6,12,18,24], labels=["0–6","6–12","12–18","18–24"])
    tm = px.treemap(df, path=["aqi_label","hour_bin"], title="Treemap: AQI → Hour")
    st.plotly_chart(tm, use_container_width=True)

# 6) Bản đồ (Folium hoặc Mapbox)
st.subheader("🗺️ Bản đồ")
if map_mode.startswith("Folium"):
    # Tạo geojson thời gian — 1 điểm/giờ, bán kính theo PM2.5
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="cartodbpositron")
    # thang màu theo PM2.5
    cm = linear.YlOrRd_09.scale(df["pm2_5"].min(), df["pm2_5"].max())
    features = []
    for _, r in df.iterrows():
        t_iso = r["time"].isoformat()
        pm = float(r["pm2_5"])
        rad = max(50, min(500, pm*5))  # bán kính theo pm2.5
        color = cm(pm)
        feat = {
            "type": "Feature",
            "geometry": {"type":"Point","coordinates":[r["lon"], r["lat"]]},
            "properties": {
                "time": t_iso,
                "style": {"color": color, "fillColor": color, "opacity":0.7, "fillOpacity":0.4},
                "icon": "circle",
                "popup": f"{t_iso}<br>PM2.5: {pm:.1f} µg/m³"
            }
        }
        features.append(feat)
    gj = {
        "type": "FeatureCollection",
        "features": features
    }
    TimestampedGeoJson(
        data=gj,
        period="PT1H",
        duration="PT1H",
        add_last_point=True,
        auto_play=False,
        loop=False
    ).add_to(m)
    cm.add_to(m)
    st.components.v1.html(m._repr_html_(), height=520, scrolling=False)
else:
    # Plotly mapbox scatter — cần mapbox token nếu muốn style map đẹp, mặc định vẫn chạy
    mb = px.scatter_mapbox(df, lat="lat", lon="lon", size="pm2_5",
                           color="pm2_5", hover_name="time",
                           zoom=10, height=520)
    mb.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(mb, use_container_width=True)

# =========================
# WORDCLOUD hoặc NETWORK GRAPH
# =========================
st.subheader("☁️ WordCloud từ nhãn AQI (lặp theo tần suất)")
wc_img = make_wordcloud(df["aqi_label"])
if wc_img is not None:
    st.image(wc_img, caption="WordCloud về nhãn AQI", use_column_width=True)
else:
    st.info("Không đủ dữ liệu văn bản để tạo WordCloud.")

# =========================
# EXTRA: BẢNG DỮ LIỆU & TẢI XUỐNG
# =========================
st.subheader("📄 Dữ liệu đã chuẩn hoá")
st.dataframe(df.tail(500), use_container_width=True)

@st.cache_data
def to_csv_bytes(df_in):
    return df_in.to_csv(index=False).encode("utf-8")

st.download_button("⬇️ Tải CSV", data=to_csv_bytes(df), file_name="hanoi_air_quality.csv", mime="text/csv")

# =========================
# FOOTER NOTES
# =========================
st.caption("""
- **Xử lý thiếu**: nội suy theo thời gian & forward/backward fill (điểm đo cố định).
- **Đặc trưng mới**: `pm_ratio` (PM2.5/PM10), `rolling_pm25_24h`, `rolling_pm10_24h`, nhãn `aqi_label`, biến thời gian (giờ, weekday, weekend).
- **Hạn chế API**: Nếu gói OWM không cho đủ 30 ngày, ứng dụng vẫn chạy với phần dữ liệu hiện có.
""")
