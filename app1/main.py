from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import joblib
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

app = FastAPI()

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Mount /static to serve report images
app.mount("/static", StaticFiles(directory=REPORTS_DIR), name="static")

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
    html += '<img src="/static/data_drift_plot.png" alt="Data Drift Plot">'
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
    return JSONResponse(status_code=404, content={"error": "Report not found"})
