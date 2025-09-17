'''from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import joblib
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

from starlette.responses import Response
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

class CORSEnabledStaticFiles(StarletteStaticFiles):
    async def get_response(self, path, scope):
        response: Response = await super().get_response(path, scope)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response





app = FastAPI()

@app.get("/")
async def monitoring_root():
    return {"message": "Welcome to the Monitoring API"}


# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Mount /static to serve report images
# Mount /static with CORS-enabled static files
app.mount("/static", CORSEnabledStaticFiles(directory=REPORTS_DIR), name="static")


# Load model
model = joblib.load(MODEL_PATH)


def save_drift_plot(drift_results):
    features = [item["feature"] for item in drift_results]
    p_values = [item["p_value"] for item in drift_results]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(features, p_values, color='green')
    plt.axhline(0.05, color='blue', linestyle='--', label='Drift threshold (p=0.05)')
    plt.title("Feature Data Drift Detection")
    plt.ylabel("p-value (KS Test)")
    plt.legend()

    plot_path = os.path.join(REPORTS_DIR, "data_drift_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


@app.get("/run-monitoring")
def run_monitoring():
    data = pd.DataFrame({
        "income": [35000, 42000, 52000, 29000, 61000, 48000, 33000, 58000, 44000, 39000],
        "credit_score": [650, 700, 720, 580, 760, 690, 640, 750, 710, 660],
        "gender": [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        "employment_status": [0, 2, 0, 3, 0, 0, 1, 2, 0, 1],
        "loan_approved": [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    })

    reference = data.sample(frac=0.6, random_state=42)
    current = data.drop(reference.index)

    X_ref = reference.drop(columns=["loan_approved"])
    X_cur = current.drop(columns=["loan_approved"])

    reference["prediction"] = model.predict(X_ref)
    current["prediction"] = model.predict(X_cur)

    # Drift detection
    drift_results = []
    for col in X_ref.columns:
        stat, p_val = ks_2samp(reference[col], current[col])
        drift_detected = p_val < 0.05
        drift_results.append({
            "feature": col,
            "p_value": round(p_val, 4),
            "drift": drift_detected
        })

    # Save plot
    save_drift_plot(drift_results)

    # Fairness
    male_preds = reference[reference["gender"] == 1]["prediction"]
    female_preds = reference[reference["gender"] == 0]["prediction"]
    mean_diff = round(male_preds.mean() - female_preds.mean(), 4)

    fairness = {
        "metric": "Mean Approval Difference (M-F)",
        "value": mean_diff
    }

    # Generate HTML report
    html = "<html><head><title>Monitoring Report</title></head><body>"
    html += "<h2>Data Drift Results </h2><ul>"
    for r in drift_results:
        html += f"<li>{r['feature']}: {'DRIFT' if r['drift'] else 'NO DRIFT'} (p={r['p_value']})</li>"
    html += "</ul>"

    html += f"<h3>Fairness Metrics:</h3><p>{fairness['metric']}: {fairness['value']}</p>"

    html += "<h2>Drift Plot</h2>"
    html += '<img src="/monitoring/static/data_drift_plot.png" alt="Data Drift Plot">'
    html += "</body></html>"

    with open(os.path.join(REPORTS_DIR, "monitoring_report.html"), "w", encoding="utf-8") as f:
        f.write(html)

    return JSONResponse(content={"status": "Monitoring completed", "fairness": fairness})


@app.get("/report-html", response_class=HTMLResponse)
def get_html_report():
    report_path = os.path.join(REPORTS_DIR, "monitoring_report.html")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>Report not found.</h3>", status_code=404)


@app.get("/report")
def get_report_file():
    path = os.path.join(REPORTS_DIR, "monitoring_report.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return JSONResponse(status_code=404, content={"error": "Report not found"})'''


from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import json
import os
import random

app = FastAPI(title="Monitoring API")

# ----------------------------
# Mount the reports folder
# ----------------------------
# This will serve files in C:/Users/Admin/credit/reports at /reports/ URL
app.mount("/reports", StaticFiles(directory="C:/Users/Admin/credit/reports"), name="reports")

# ----------------------------
# Path to results JSON
# ----------------------------
RESULTS_FILE = "C:/Users/Admin/credit/reports/monitoring_results.json"


# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "Monitoring Root"}


# ----------------------------
# Run Monitoring
# ----------------------------
@app.get("/run-monitoring")
def run_monitoring():
    """
    Simulate monitoring run.
    Replace this with actual drift, fairness, and performance calculations.
    """
    # Simulated data drift
    data_drift = {
        "income": {"drift": "NO DRIFT", "p_value": round(random.uniform(0.2, 0.8), 4)},
        "credit_score": {"drift": "NO DRIFT", "p_value": round(random.uniform(0.2, 0.8), 4)},
        "gender": {"drift": "NO DRIFT", "p_value": 1.0},
        "employment_status": {"drift": "NO DRIFT", "p_value": round(random.uniform(0.2, 0.8), 4)},
    }

    # Fairness metrics
    fairness = {"mean_approval_diff": round(random.uniform(0.2, 0.5), 4)}

    # Model quality metrics
    model_quality = {
        "accuracy": round(random.uniform(0.8, 0.95), 4),
        "precision": round(random.uniform(0.75, 0.9), 4),
        "recall": round(random.uniform(0.7, 0.9), 4),
        "f1_score": round(random.uniform(0.7, 0.9), 4),
    }

    # Operations metrics
    operations = {
        "latency_ms": random.randint(80, 250),
        "throughput_rps": random.randint(50, 150),
    }

    # Dynamic suggestions
    suggestions = []
    if model_quality["accuracy"] < 0.85 or model_quality["f1_score"] < 0.8:
        suggestions.append("Consider retraining the model with more recent or balanced data.")
    if fairness["mean_approval_diff"] > 0.3:
        suggestions.append("Investigate fairness issues; consider reweighting or adding fairness constraints.")
    if operations["latency_ms"] > 200:
        suggestions.append("Optimize inference pipeline to reduce latency (e.g., batch processing).")
    if operations["throughput_rps"] < 50:
        suggestions.append("Increase throughput by scaling API or optimizing model serving.")
    if len(suggestions) == 0:
        suggestions.append("All metrics are within acceptable range. Continue monitoring.")

    # Path for drift plot
    drift_plot_path = "/reports/data_drift_plot.png"

    results = {
        "data_drift": data_drift,
        "fairness": fairness,
        "model_quality": model_quality,
        "operations": operations,
        "suggestions": suggestions,
        "drift_plot": drift_plot_path
    }

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    return {"message": "Monitoring run completed.", "file": RESULTS_FILE}


# ----------------------------
# Get JSON report (inline)
# ----------------------------
@app.get("/report")
def get_report():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content={"error": "No report found. Run /run-monitoring first."}, status_code=404)


# ----------------------------
# Get HTML report (dashboard)
# ----------------------------
@app.get("/report-html", response_class=HTMLResponse)
def get_report_html():
    try:
        if not os.path.exists(RESULTS_FILE):
            return HTMLResponse("<h3>No report found. Run /run-monitoring first.</h3>", status_code=404)

        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)

        # Safely get values
        data_drift = results.get("data_drift", {})
        fairness = results.get("fairness", {})
        model_quality = results.get("model_quality", {})
        operations = results.get("operations", {})
        suggestions = results.get("suggestions", [])
        drift_plot = results.get("drift_plot", "")

        # Build HTML blocks
        drift_html = "".join(
            f"<div class='card'><h3>{feature}</h3><p>{info.get('drift','N/A')} (p={info.get('p_value','N/A')})</p></div>"
            for feature, info in data_drift.items()
        )

        quality_html = "".join(
            f"<div class='card'><h3>{metric}</h3><p>{value}</p></div>"
            for metric, value in model_quality.items()
        )

        operations_html = "".join(
            f"<div class='card'><h3>{metric}</h3><p>{value}</p></div>"
            for metric, value in operations.items()
        )

        suggestions_html = "".join(f"<li>{s}</li>" for s in suggestions)

        html_content = f"""
        <html>
        <head><title>ML- Observability Monitoring Dashboard</title>
        <style>
        body {{ font-family: Arial; background:#f9f9f9; padding:20px; }}
        h1 {{ text-align:center; color:#333; }}
        h2 {{ margin-top:40px; color:#444; }}
        .container {{ display:flex; flex-wrap:wrap; gap:15px; }}
        .card {{ background:#fff; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1); flex:1 1 200px; text-align:center; }}
        ul {{ background:#fff; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1); }}
        img {{ display:block; margin:auto; max-width:90%; }}
        </style>
        </head>
        <body>
        <h1>ML- Observability Monitoring Dashboard</h1>

        <h2>Data Drift Results</h2><div class="container">{drift_html}</div>

        <h2>Fairness Metrics</h2>
        <div class="container"><div class="card"><h3>Mean Approval Difference (M-F)</h3><p>{fairness.get('mean_approval_diff','N/A')}</p></div></div>

        <h2>Model Quality</h2><div class="container">{quality_html}</div>

        <h2>Operations</h2><div class="container">{operations_html}</div>

        <h2>Suggestions to Improve</h2><ul>{suggestions_html}</ul>

        <h2>Drift Plot</h2><div class="container">
        {"<img src='"+drift_plot+"' alt='Data Drift Plot'/>" if drift_plot else "<p>No plot available</p>"}
        </div>

        </body></html>
        """

        return HTMLResponse(content=html_content)

    except Exception as e:
        return HTMLResponse(f"<h3>Error generating dashboard: {e}</h3>", status_code=500)
