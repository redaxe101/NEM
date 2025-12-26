from flask import Flask, request, jsonify

# import tensorflow as tf
import numpy as np
import os
import joblib
from pathlib import Path
from zoneinfo import ZoneInfo
from flask import render_template_string
import threading
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.expand_frame_repr", False)

decoder_feature_cols = [
    "F_RRP_NSW1",
    "F_RRP_VIC1",
    "F_RRP_QLD1",
    "F_RRP_SA1",
    "F_RRP_TAS1",
    "F_TOTALDEMAND_NSW1",
    "F_TOTALDEMAND_VIC1",
    "F_TOTALDEMAND_QLD1",
    "F_TOTALDEMAND_SA1",
    "F_TOTALDEMAND_TAS1",
    "F_rrp_min_past_NSW1",
    "F_rrp_min_past_VIC1",
    "F_rrp_min_past_QLD1",
    "F_rrp_zscore_NSW1",
    "F_rrp_zscore_QLD1",
    "F_rrp_zscore_VIC1",
    "F_rrp_zscore_TAS1",
    "F_rrp_zscore_SA1",
    "F_half_hour_sin",
    "F_half_hour_cos",
    "F_hours_to_delivery",
    "F_TEMPHUMIDITY",
    "F_TEMP_ABOVE",
    "F_TEMP_BELOW",
    "F_rrp_to_demand_NSW1",
    "F_rrp_to_demand_QLD1",
    "F_rrp_to_demand_VIC1",
]
encoder_feature_cols = [
    "RRP_NSW1",
    "RRP_VIC1",
    "RRP_QLD1",
    "RRP_TAS1",
    "RRP_SA1",
    "TEMPHUMIDITY",
    "TEMP_ABOVE",
    "TEMP_BELOW",
    "rrp_to_demand_NSW1",
    "rrp_to_demand_VIC1",
    "rrp_to_demand_QLD1",
    "rrp_max_past_NSW1",
    "rrp_max_past_VIC1",
    "rrp_max_past_QLD1",
    "rrp_skew_NSW1",
    "rrp_skew_QLD1",
    "rrp_skew_VIC1",
    "rrp_kurt_NSW1",
    "rrp_kurt_QLD1",
    "rrp_kurt_VIC1",
]

global_regions = ["NSW1", "VIC1", "QLD1"]

latest_timestamp = None
previous_timestamp = None

timestamps = None
input_length = 8
output_length = 32

dec_scaler = {}
enc_scaler = {}
rrp_scaler = {}
dem_scaler = {}
model = {}
latest_rrp = {}
current_rrp = {}
current_timestamp = {}
latest_spike_prob = {}
latest_dem = {}
previous_rrp = {}

for region in global_regions:
    dec_scaler[region] = joblib.load(
        os.path.join(
            os.path.dirname(__file__), f"dec_scaler_i{input_length}_{region}.joblib"
        )
    )
    enc_scaler[region] = joblib.load(
        os.path.join(
            os.path.dirname(__file__), f"enc_scaler_i{input_length}_{region}.joblib"
        )
    )
    rrp_scaler[region] = joblib.load(
        os.path.join(
            os.path.dirname(__file__), f"rrp_scaler_i{input_length}_{region}.joblib"
        )
    )
    dem_scaler[region] = joblib.load(
        os.path.join(
            os.path.dirname(__file__), f"dem_scaler_i{input_length}_{region}.joblib"
        )
    )
    model[region] = None
    latest_rrp[region] = None
    current_rrp[region] = None
    latest_spike_prob[region] = None
    latest_dem[region] = None
    previous_rrp[region] = None

app = Flask(__name__)
# Load model at startup instead of using before_first_request


def get_model(region):
    global model
    model_path = os.path.join(
        os.path.dirname(__file__),
        f"transformer_model_i{input_length}_small_{region}.keras",
    )
    if model[region] is None:
        import tensorflow as tf

        print("⏳ Loading model...")
        model[region] = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully.")
    return model[region]


def add_time_features(df, region):
    """
    Adds sine and cosine time-based features to a DataFrame with a DatetimeIndex or MultiIndex,
    using 30-minute intervals that are adjusted for daylight savings.

    Features:
        - half_hour_sin, half_hour_cos (48 values/day, DST-adjusted)
        - dow_sin, dow_cos (day of week)
        - month_sin, month_cos (month of year)
    """
    import pandas as pd
    import numpy as np

    df = df.copy()

    if region == "QLD1":
        timezone = "Australia/Brisbane"
    else:
        timezone = "Australia/Sydney"

    if not isinstance(df.index, (pd.DatetimeIndex, pd.MultiIndex)):
        raise ValueError("df must have a DatetimeIndex or MultiIndex")

    # Extract datetime index (convert to local time with DST handling)
    if isinstance(df.index, pd.MultiIndex):
        datetime_index = df.index.get_level_values("DATETIME")
    else:
        datetime_index = df.index

    # subtract 30 minutes to align with beginning of half-hour period
    datetime_index = pd.to_datetime(datetime_index) - pd.Timedelta(minutes=30)

    # Ensure timezone-aware index
    if datetime_index.tz is None:
        datetime_index = datetime_index.tz_localize("UTC").tz_convert(timezone)
    else:
        datetime_index = datetime_index.tz_convert(timezone)

    # Half hour slot
    half_hour_slot = datetime_index.hour * 2 + (datetime_index.minute // 30)
    df["half_hour_sin"] = np.sin(2 * np.pi * half_hour_slot / 48)
    df["half_hour_cos"] = np.cos(2 * np.pi * half_hour_slot / 48)

    # Day of week
    df["dow_sin"] = np.sin(2 * np.pi * datetime_index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * datetime_index.dayofweek / 7)
    df["workday"] = (datetime_index.dayofweek <= 4).astype(int)

    # Month of year
    df["month_sin"] = np.sin(2 * np.pi * datetime_index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * datetime_index.month / 12)

    return df


def add_other_features(df):
    df = df.copy()
    df["rrp_yesterday"] = df["RRP"].shift(48)
    df["rrp_rolling_std"] = df["RRP"].rolling(4).std().fillna(0)
    df["rrp_max_past"] = df["RRP"].rolling(4).max().fillna(0)
    df["rrp_max_past_day"] = df["RRP"].rolling(48).max().fillna(0)
    df["rrp_pct_change"] = (
        df["RRP"].pct_change(fill_method=None).fillna(0).clip(lower=-1000, upper=1000)
    )
    df["rrp_abs_change"] = df["RRP"].diff().abs().fillna(0)
    df["rrp_min_past"] = df["RRP"].rolling(4).min().fillna(0)
    df["rrp_range"] = df["rrp_max_past"] - df["rrp_min_past"]
    df["rrp_rolling_quantile_90"] = df["RRP"].rolling(96).quantile(0.9).fillna(0)
    df["rrp_above_q90"] = (df["RRP"] > df["rrp_rolling_quantile_90"]).astype(int)
    df["demand_rolling_quantile_90"] = (
        df["TOTALDEMAND"].rolling(96).quantile(0.9).fillna(0)
    )
    df["demand_above_q90"] = (
        df["TOTALDEMAND"] > df["demand_rolling_quantile_90"]
    ).astype(int)
    df["rrp_skew"] = df["RRP"].rolling(24).skew().fillna(0)
    df["rrp_kurt"] = df["RRP"].rolling(24).kurt().fillna(0)
    df["rrp_ma_12"] = df["RRP"].rolling(12).mean().fillna(0)
    df["rrp_dev_ma12"] = (df["RRP"] - df["rrp_ma_12"]) / (
        df["rrp_ma_12"].abs() + 1e-6
    ).clip(lower=0, upper=10)
    window = 24
    df["rrp_mean_24"] = df["RRP"].rolling(window).mean()
    df["rrp_std_24"] = df["RRP"].rolling(window).std()
    df["rrp_zscore"] = (df["RRP"] - df["rrp_mean_24"]) / (df["rrp_std_24"] + 1e-6).clip(
        lower=-10, upper=10
    )
    df["rrp_to_demand"] = (df["RRP"] / (df["TOTALDEMAND"] + 1e-6)).clip(
        lower=0, upper=1
    )
    # df["AVAILABLEGENERATION"] = (
    #     df["SCHEDULEDGENERATION"]
    #     + df["SEMISCHEDULEDGENERATION"]
    # )
    # df["RESERVE_MARGIN"] = df["AVAILABLEGENERATION"] - df["TOTALDEMAND"]
    # df["RENEWABLES_SHARE"] = df["SEMISCHEDULEDGENERATION"] / (df["AVAILABLEGENERATION"])
    # df["IMPORT_RATIO"] = df["NETINTERCHANGE"] / df["TOTALDEMAND"]
    return df


def build_encoder_input_from_aemo(df, weather_df, region):
    global previous_rrp, previous_timestamp

    enc_df = df[df["PERIODTYPE"] == "ACTUAL"]

    enc_df = add_time_features(enc_df, region)
    enc_df = enc_df.join(weather_df, how="left")
    enc_df = enc_df.reindex(columns=encoder_feature_cols, fill_value=0.0)
    enc_df = enc_df[-input_length:]

    previous_rrp[region] = enc_df[f"RRP_{region}"].values.tolist()[-12:]
    previous_timestamp = enc_df.index.tolist()[-12:]
    scaled = enc_scaler[region].transform(np.array(enc_df.fillna(0), dtype=np.float32))

    encoder_input = np.expand_dims(
        scaled[:, : len(encoder_feature_cols)], axis=0
    ).astype(np.float32)

    return encoder_input


def build_decoder_input_from_aemo(df, weather_df, region):
    dec_df = df[df["PERIODTYPE"] == "FORECAST"]

    dec_df = add_time_features(dec_df, region)
    df_index = dec_df.index
    dec_df = dec_df.join(weather_df, how="left")
    dec_df["hours_to_delivery"] = (
        dec_df.index - dec_df.index[0]
    ).total_seconds() / 3600

    dec_df = dec_df.add_prefix("F_")
    dec_df = dec_df.reindex(columns=decoder_feature_cols, fill_value=0.0)
    dec_df = dec_df[:output_length]
    scaled = dec_scaler[region].transform(np.array(dec_df.fillna(0), dtype=np.float32))

    decoder_input = np.expand_dims(
        scaled[:, : len(decoder_feature_cols)], axis=0
    ).astype(np.float32)
    return decoder_input, df_index


def fetch_actual_weather(region):
    """
    Fetch actual weather data from Open-Meteo API
    """
    print(f"🌤 Fetching weather ...")

    match region:
        case "NSW1":
            lat = -33.8148
            lon = 151.0017
        case "QLD1":
            lat = -27.4705
            lon = 153.0251
        case "VIC1":
            lat = -37.8136
            lon = 144.9631
        case "TAS1":
            lat = -42.8829
            lon = 147.3272
        case "SA1":
            lat = -34.9285
            lon = 138.5999

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,cloudcover,relative_humidity_2m,windspeed_10m",
        "timezone": "Australia/Brisbane",
    }
    r = requests.get(url, params=params)
    data = r.json()["hourly"]
    new_df = pd.DataFrame(data)
    new_df["time"] = pd.to_datetime(new_df["time"])
    new_df.set_index("time", inplace=True)
    new_df = new_df.resample("30min").interpolate("linear").ffill().bfill()
    new_df["TEMPHUMIDITY"] = new_df["temperature_2m"] * new_df["relative_humidity_2m"]
    new_df["TEMP_ABOVE"] = (new_df["temperature_2m"] - 27).clip(lower=0)
    new_df["TEMP_BELOW"] = (16 - new_df["temperature_2m"]).clip(lower=0)
    new_df.index = new_df.index.tz_localize("Australia/Brisbane")

    return new_df


def fetch_and_predict_loop():
    global current_rrp, current_timestamp, latest_rrp, latest_spike_prob, latest_dem, latest_timestamp, timestamps
    global model

    last_weather = {}
    for region in global_regions:
        model[region] = get_model(region)
        last_weather[region] = None
    while True:
        try:

            print("🔁 Fetching AEMO data and running predictions...")

            response = requests.post(
                "https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN",
                json={"timeScale": ["30MIN"]},
                timeout=10,
            )
            response.raise_for_status()
            forecasts_raw = response.json()

            if "5MIN" not in forecasts_raw:
                raise ValueError("Missing '5MIN' key in AEMO response")

            forecasts = forecasts_raw["5MIN"]
            df = pd.DataFrame(forecasts)
            df["SETTLEMENTDATE"] = pd.to_datetime(
                df["SETTLEMENTDATE"], format="%Y-%m-%dT%H:%M:%S"
            )
            df = df.set_index("SETTLEMENTDATE")
            df.index = df.index.tz_localize("Australia/Brisbane")
            # first_row = df.iloc[[0]].copy()
            # first_row.index = first_row.index - pd.Timedelta(minutes=30)
            # df = pd.concat([first_row, df]).sort_index()
            for region in global_regions:

                if not last_weather[region] or datetime.now(
                    tz=ZoneInfo("UTC")
                ) - last_weather[region] > timedelta(minutes=60):
                    weather_df = fetch_actual_weather(region)
                    last_weather[region] = datetime.now(tz=ZoneInfo("UTC"))

                main_df = df[df["REGION"] == region]

                idx = main_df.loc[main_df["PERIODTYPE"] == "ACTUAL"].index.max()
                current_rrp[region] = main_df.loc[idx, "RRP"]
                current_timestamp[region] = idx

                main_df = main_df.resample("30min", label="right", closed="right").agg(
                    {
                        **{col: "mean" for col in df.select_dtypes("number").columns},
                        **{
                            col: "last"
                            for col in df.select_dtypes(exclude="number").columns
                        },
                    }
                )
                for sample_region in ("NSW1", "VIC1", "QLD1", "TAS1", "SA1"):
                    new_df = df[df["REGION"] == sample_region]
                    new_df = add_other_features(new_df)
                    new_df = new_df.add_suffix("_" + sample_region)
                    main_df = main_df.join(new_df)

                decoder_input, timestamps = build_decoder_input_from_aemo(
                    main_df, weather_df, region
                )
                encoder_input = build_encoder_input_from_aemo(
                    main_df, weather_df, region
                )

                preds_scaled = model[region].predict([encoder_input, decoder_input])

                latest_rrp[region] = (
                    rrp_scaler[region]
                    .inverse_transform(preds_scaled[0].reshape(-1, 1))
                    .tolist()
                )
                latest_dem[region] = (
                    dem_scaler[region]
                    .inverse_transform(preds_scaled[1].reshape(-1, 1))
                    .tolist()
                )
                latest_spike_prob[region] = preds_scaled[2].reshape(-1, 1).tolist()

            latest_timestamp = datetime.now(tz=ZoneInfo("UTC"))

            print("✅ Prediction updated at", latest_timestamp)

        except Exception as e:
            print("❌ Error in prediction loop:", e)

        time.sleep(((15 - (time.time() % 60)) % 60) or 60)


def last_saved_model_date(model_path: str, tz="UTC") -> str:
    dt = datetime.fromtimestamp(Path(model_path).stat().st_mtime, ZoneInfo(tz))
    return f"{dt.day} {dt:%B %Y}"


@app.route("/")
def index():
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>NEM Forecast</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; color: #333; }}
            footer {{ margin-top: 2em; font-size: 0.85em; color: #777; border-top: 1px solid #ccc; padding-top: 1em; }}
            a {{ color: #0066cc; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        <h2>NEM Forecast - Deep Learning Transformer Models</h2>
        <div style="display: flex; align-items: center; max-width: 500px;">
            <img src="/static/une_logo.png" alt="UNE Logo"
                style="width: 60px; height: auto; margin-right: 10px;">
            <div style="line-height: 1.4;">
                Developed by <b>Mark Sinclair</b>,<br>
                University of New England, 2025<br>
                <a href='https://orcid.org/0009-0004-0702-8193'>ORCiD</a>
            </div>
        </div>

        <div style="max-width: 600px;">
            <p>Deep learning transformer models for forecasting electricity spot prices, spike probabilities, and demand in the Australian National Electricity Market (NEM).
            Based on the paper <a href="https://www.mdpi.com/2076-3417/16/1/75" target="_blank">"Learning the Grid: Transformer Architectures for Electricity Price Forecasting in the Australian National Market"</a></p>
            <p>
                NSW1 <a href="/predict/NSW1">API</a> | <a href="/chart/NSW1">Chart</a>
            </p>
            <p>
                VIC1 <a href="/predict/VIC1">API</a> | <a href="/chart/VIC1">Chart</a>
            </p>
            <p>
                QLD1 <a href="/predict/QLD1">API</a> | <a href="/chart/QLD1">Chart</a>
            </p>
        </div>

        <footer style="max-width: 600px;text-align: justify;">
        © 2025 <a href="https://www.linkedin.com/in/markwsinclair/" target="_blank">Mark Sinclair</a>. Developed as part of postgraduate research at the 
        <a href="https://www.une.edu.au" target="_blank">University of New England</a>, Australia. 
        Forecasts and model outputs are licensed under 
        <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a> — 
        free for use and redistribution, including commercial use, with attribution. 
        Based on data © Australian Energy Market Operator (AEMO) and © Open-Meteo Weather API, licensed under 
        <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a>.
        </footer>
    </body>
    </html>
    """

    return html, 200, {"Content-Type": "text/html"}


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/chart/<region>", methods=["GET"])
def chart(region):
    return render_template_string(
        """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>NEM Forecasts</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <!-- Chart.js + Luxon time adapter -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1"></script>
    <style>
      :root { color-scheme: dark; }
      body { margin:0; padding:2rem; font-family:system-ui, sans-serif; background:#0f0f0f; color:#ddd; }
      #wrap { max-width: 1100px; margin: 0 auto; }
      /* Responsive canvas: give it a fixed height while letting width be responsive */
      .chart-wrap { position: relative; height: 420px; }
      canvas { background:#1b1b1b; border-radius:12px; }
      small { color:#aaa; }
      #err { color:#f66; white-space:pre-wrap; margin-top:.75rem; }
        footer { margin-top: 2em; font-size: 0.85em; color: #777; border-top: 1px solid #ccc; padding-top: 1em; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
  </head>
  <body>
    <div id="wrap">
        <h2 id="title">NEM Forecast</h2>
        <small id="stamp"></small>
        <div class="chart-wrap">
            <canvas id="chart"></canvas>
        </div>
        <div id="err"></div>

        <footer style="text-align: justify;">
        © 2025 <a href="https://www.linkedin.com/in/markwsinclair/" target="_blank">Mark Sinclair</a>. Developed as part of postgraduate research at the 
        <a href="https://www.une.edu.au" target="_blank">University of New England</a>, Australia. 
        Forecasts and model outputs are licensed under 
        <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a> — 
        free for use and redistribution, including commercial use, with attribution. 
        Based on data © Australian Energy Market Operator (AEMO) and © Open-Meteo Weather API, licensed under 
        <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a>.
        </footer>   
    </div>

    <script>
    (async () => {
      const el   = document.getElementById("chart");
      const h2   = document.getElementById("title");
      const err  = document.getElementById("err");
      const stamp= document.getElementById("stamp");

      // ISO -> epoch ms via Luxon (robust across zones)
      const parseTime = iso => {
        const dt = luxon.DateTime.fromISO(iso);
        return dt.isValid ? dt.toMillis() : NaN;
      };
      const toXY = (ts, ys) => {
        const n = Math.min(ts.length, ys.length);
        const out = [];
        for (let i = 0; i < n; i++) {
          const x = parseTime(ts[i]);
          const y = Number(ys[i]);
          if (Number.isFinite(x) && Number.isFinite(y)) out.push({ x, y });
        }
        return out;
      };

      async function load() {
        const r = await fetch(`/predict/{{ region }}`, { cache: "no-store" });
        if (!r.ok) throw new Error("HTTP " + r.status);
        const j = await r.json();
        const region = j.region || "Unknown";
        document.title = `NEM Prediction — ${region}`;
        h2.textContent = `NEM Prediction (${region})`;
        stamp.textContent = "Prediction time: " + (j.predictionTime || "").slice(0, 16);

        return {
          price:  toXY(j.nemTimestamp ?? [], j.spotPrice ?? []),
          spike:  toXY(j.nemTimestamp ?? [], j.spikeProbability ?? []), 
          demand: toXY(j.nemTimestamp ?? [], j.totalDemand ?? []),
          prev:   toXY((j.previousTimestamp ?? j.nemTimestamp) ?? [],
                 (j.previousRrp ?? j.previous_rrp ?? [])),
              };
      }

      try {
        const d = await load();

        const chart = new Chart(el, {
          type: "line",
          data: {
            datasets: [
              { label: "Prev RRP (history)", data: d.prev,   yAxisID: "y1",
                    borderColor: "#a855f7", borderDash: [4,3], pointRadius: 0, tension: 0, spanGaps: true },
              { label: "Spot Price (A$/MWh)", data: d.price,  yAxisID: "y1",
                borderColor: "#4ade80", borderWidth: 2, pointRadius: 0, tension: 0.2, spanGaps: true },
              { label: "Spike Prob.",         data: d.spike,  yAxisID: "y2",
                borderColor: "#60a5fa", borderDash: [6,4], pointRadius: 0, tension: 0.2, spanGaps: true },
              { label: "Demand (MW)",         data: d.demand, yAxisID: "y3",
                hidden: true,
                borderColor: "#f59e0b", pointRadius: 0, tension: 0.2, spanGaps: true }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,  // use CSS height
            parsing: false,              // we supply {x,y}
            animation: false,
            interaction: { mode: "nearest", intersect: false },
            scales: {
              x: {
                type: "time",
                time: {
                  unit: "hour",
                  displayFormats: { minute: "HH:mm", hour: "HH:mm" },
                  tooltipFormat: "ccc HH:mm"     // Mon 13:30
                }
              },
              y1: { position: "left", suggestedMin: 0, suggestedMax: 350,
                    title: { display: true, text: "A$/MWh" } },
              y2: { position: "right", min: 0, max: 1,
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: "Spike prob." } },
              y3: { position: "right", display: false,
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: "MW" } }
            },
            plugins: {
              legend: { position: "bottom" },
              tooltip: {
                callbacks: {
                  label: (c) => {
                    const y = c.parsed.y;
                    if (c.dataset.yAxisID === "y2") return ` ${y.toFixed(3)} prob`;
                    if (c.dataset.yAxisID === "y3") return ` ${Math.round(y).toLocaleString()} MW`;
                    return ` $${y.toFixed(2)} / MWh`;
                  }
                }
              }
            },
            adapters: { date: { zone: "utc" } }
          }
        });

        // Refresh every minute (match your background cadence)
        setInterval(async () => {
          try {
            const d2 = await load();
            chart.data.datasets[0].data = d2.prev; 
            chart.data.datasets[1].data = d2.price;
            chart.data.datasets[2].data = d2.spike;
            chart.data.datasets[3].data = d2.demand;
            chart.update("none");
          } catch (e) {
            console.error("Refresh failed:", e);
          }
        }, 60_000);

      } catch (e) {
        console.error(e);
        err.textContent = "Error: " + (e?.message || e);
      }
    })();
    </script>

  </body>
</html>
""",
        region=region,
    )


def _flat(xs):
    return [float(v[0]) if isinstance(v, (list, tuple)) else float(v) for v in xs]


@app.route("/predict/<region>", methods=["GET"])
def predict(region):
    if latest_rrp[region] is None:
        return jsonify({"error": "Prediction not ready yet"}), 503

    return jsonify(
        {
            "nemTimestamp": [ts.isoformat() for ts in timestamps],
            "predictionTime": (
                latest_timestamp.isoformat() if latest_timestamp else None
            ),
            "spotPrice": _flat(latest_rrp[region]),
            "spikeProbability": _flat(latest_spike_prob[region]),
            "totalDemand": _flat(latest_dem[region]),
            "region": region,
            "previousRrp": previous_rrp[region],
            "previousTimestamp": [ts.isoformat() for ts in previous_timestamp],
            "currentRrp": current_rrp[region],
            "currentTimestamp": current_timestamp[region].isoformat(),
        }
    )


threading.Thread(target=fetch_and_predict_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
