from flask import Flask, request, jsonify
import tensorflow as tf
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
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

decoder_feature_cols = [
    "F_TOTALDEMAND",
    "F_RRP",
    "F_AVAILABLEGENERATION",
    "F_RESERVE_MARGIN",
    "F_RENEWABLES_SHARE",
    "F_IMPORT_RATIO",
    "F_TOTALDEMAND_NSW1",
    "F_RRP_NSW1",
    "F_TOTALDEMAND_VIC1",
    "F_RRP_VIC1",
    "F_TOTALDEMAND_QLD1",
    "F_RRP_QLD1",
    "F_AVAILABLEGENERATION_VIC1",
    "F_RESERVE_MARGIN_VIC1",
    "F_RENEWABLES_SHARE_VIC1",
    "F_IMPORT_RATIO_VIC1",
    "F_AVAILABLEGENERATION_QLD1",
    "F_RESERVE_MARGIN_QLD1",
    "F_RENEWABLES_SHARE_QLD1",
    "F_IMPORT_RATIO_QLD1",
    "F_half_hour_sin",
    "F_half_hour_cos",
    "F_dow_sin",
    "F_dow_cos",
    "F_month_sin",
    "F_month_cos",
    "F_temperature_2m",
    "F_cloudcover",
    "F_relative_humidity_2m",
    "F_windspeed_10m",
    "F_TEMPHUMIDITY",
    "F_TEMP_ABOVE_28",
    "F_TEMP_BELOW_16",
    "F_workday",
]
encoder_feature_cols = [
    "RRP",
    "TOTALDEMAND",
    "RRP_NSW1",
    "TOTALDEMAND_NSW1",
    "RRP_VIC1",
    "TOTALDEMAND_VIC1",
    "RRP_QLD1",
    "TOTALDEMAND_QLD1",
    "half_hour_sin",
    "half_hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "temperature_2m",
    "cloudcover",
    "relative_humidity_2m",
    "windspeed_10m",
    "TEMPHUMIDITY",
    "TEMP_ABOVE_28",
    "TEMP_BELOW_16",
    "rrp_rolling_std_1h",
    "rrp_max_past_1h",
    "workday",
]

latest_rrp = None
latest_spike_prob = None
latest_dem = None
latest_timestamp = None

timestamps = None
region = "NSW1"
input_length = 4
output_length = 32

dec_scaler = joblib.load(
    os.path.join(os.path.dirname(__file__), f"{region}_dec_scaler.joblib")
)

enc_scaler = joblib.load(
    os.path.join(os.path.dirname(__file__), f"{region}_enc_scaler.joblib")
)

rrp_scaler = joblib.load(
    os.path.join(os.path.dirname(__file__), f"{region}_rrp_scaler.joblib")
)

dem_scaler = joblib.load(
    os.path.join(os.path.dirname(__file__), f"{region}_dem_scaler.joblib")
)

app = Flask(__name__)
# Load model at startup instead of using before_first_request
model_path = os.path.join(
    os.path.dirname(__file__), f"transformer_model_small_{region}.keras"
)
model = tf.keras.models.load_model(model_path)


def build_encoder_input_from_aemo():
    # Fetch latest NEM/AEMO data yo
    # Preprocess into shape (1, 48, 14)

    encoder_input_raw = np.random.rand(1, input_length, 14)

    encoder_scaled = []
    num_features = len(encoder_feature_cols)
    for row in encoder_input_raw[0]:
        full_row = np.zeros((num_features,))
        full_row_scaled = enc_scaler.transform(full_row.reshape(1, -1))[0]
        encoder_scaled.append(full_row_scaled)
    encoder_input = np.expand_dims(np.array(encoder_scaled), axis=0).astype(np.float32)
    return encoder_input


def build_decoder_input_from_aemo():

    response = requests.post(
        "https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN",
        json={"timeScale": ["30MIN"]},
    )
    response.raise_for_status()
    forecasts_raw = response.json()

    if "5MIN" not in forecasts_raw:
        raise ValueError("Missing '5MIN' key in AEMO response")

    forecasts = forecasts_raw["5MIN"]
    forecasts = [{f"F_{k}": v for k, v in row.items()} for row in forecasts]

    df = pd.DataFrame(forecasts)
    df["F_SETTLEMENTDATE"] = pd.to_datetime(df["F_SETTLEMENTDATE"])
    df = df.set_index("F_SETTLEMENTDATE")
    df.index = df.index.tz_localize("Australia/Brisbane")
    # Filter for NSW1
    enc_df = df[df["F_REGION"] == region]
    enc_df = enc_df.resample("30min", label="right", closed="right").mean(
        numeric_only=True
    )[:output_length]
    df_index = enc_df.index

    for sample_region in ("NSW1", "VIC1", "QLD1"):
        new_df = df[df["F_REGION"] == sample_region]
        new_df = new_df.resample("30min", label="right", closed="right").mean(
            numeric_only=True
        )[:output_length]
        new_df = new_df.add_suffix("_" + sample_region)
        enc_df = enc_df.join(new_df)

    enc_df = enc_df.reindex(columns=decoder_feature_cols, fill_value=0.0)

    scaled = dec_scaler.transform(enc_df.values)

    decoder_input = np.expand_dims(
        scaled[:, : len(decoder_feature_cols)], axis=0
    ).astype(np.float32)
    return decoder_input, df_index


def fetch_and_predict_loop():
    global latest_rrp, latest_spike_prob, latest_dem, latest_timestamp, timestamps

    while True:
        try:
            print("🔁 Fetching AEMO data and running prediction...")

            # === Replace this with real AEMO fetching and input generation ===
            encoder_input = build_encoder_input_from_aemo()  # shape: (1, 48, 14)
            decoder_input, timestamps = (
                build_decoder_input_from_aemo()
            )  # shape: (1, 32, 35)

            preds_scaled = model.predict([encoder_input, decoder_input])
            # preds_scaled = preds_scaled.reshape(-1)

            latest_rrp = rrp_scaler.inverse_transform(
                preds_scaled[0].reshape(-1, 1)
            ).tolist()
            latest_dem = dem_scaler.inverse_transform(
                preds_scaled[1].reshape(-1, 1)
            ).tolist()
            latest_spike_prob = preds_scaled[2].reshape(-1, 1).tolist()

            latest_timestamp = datetime.now()

            print("✅ Prediction updated at", latest_timestamp)

        except Exception as e:
            print("❌ Error in prediction loop:", e)

        time.sleep(((15 - (time.time() % 60)) % 60) or 60)


def last_saved_model_date(model_path: str, tz="UTC") -> str:
    dt = datetime.fromtimestamp(Path(model_path).stat().st_mtime, ZoneInfo(tz))
    return f"{dt.day} {dt:%B %Y}"


@app.route("/")
def index():
    model_path = f"transformer_model.keras"
    return (
        f"NEM spot price predictor by Mark Sinclair, University of New England, 2025.<br/><br/><a href='predict'>{region}</a> <a href='/chart'>view chart</a> Model trained: {last_saved_model_date(model_path)}",
        200,
    )


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.get("/chart")
def chart():
    return render_template_string(
        """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>NEM Prediction</title>
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
    </style>
  </head>
  <body>
    <div id="wrap">
      <h2 id="title">NEM prediction</h2>
      <small id="stamp"></small>
      <div class="chart-wrap">
        <canvas id="chart"></canvas>
      </div>
      <div id="err"></div>
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
        const r = await fetch("/predict", { cache: "no-store" });
        if (!r.ok) throw new Error("HTTP " + r.status);
        const j = await r.json();

        const region = j.region || "Unknown";
        document.title = `NEM Prediction — ${region}`;
        h2.textContent = `NEM prediction (${region})`;
        stamp.textContent = "Prediction time: " + (j.predictionTime || "");

        return {
          price:  toXY(j.nemTimestamp ?? [], j.spotPrice ?? []),
          spike:  toXY(j.nemTimestamp ?? [], j.spikeProbability ?? []), // already 0..1
          demand: toXY(j.nemTimestamp ?? [], j.totalDemand ?? []),
        };
      }

      try {
        const d = await load();

        const chart = new Chart(el, {
          type: "line",
          data: {
            datasets: [
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
              y1: { position: "left",
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
            adapters: { date: { zone: "Australia/Sydney" } }
          }
        });

        // Refresh every minute (match your background cadence)
        setInterval(async () => {
          try {
            const d2 = await load();
            chart.data.datasets[0].data = d2.price;
            chart.data.datasets[1].data = d2.spike;
            chart.data.datasets[2].data = d2.demand;
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
"""
    )


def _flat(xs):
    return [float(v[0]) if isinstance(v, (list, tuple)) else float(v) for v in xs]


@app.route("/predict", methods=["GET"])
def predict():
    if latest_rrp is None:
        return jsonify({"error": "Prediction not ready yet"}), 503

    return jsonify(
        {
            "nemTimestamp": [ts.isoformat() for ts in timestamps],
            "predictionTime": latest_timestamp.isoformat(),
            "spotPrice": _flat(latest_rrp),
            "spikeProbability": _flat(latest_spike_prob),
            "totalDemand": _flat(latest_dem),
            "region": region,
        }
    )


threading.Thread(target=fetch_and_predict_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
