'''import pandas as pd
import joblib
import os
from scipy.stats import ks_2samp

# Optional fairness fallback
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False


def run_monitoring():
    # Get base directory as one level up from this file (i.e., /credit)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Sample test data
    data = pd.DataFrame({
        "income": [35000, 42000, 52000, 29000, 61000, 48000, 33000, 58000, 44000, 39000],
        "credit_score": [650, 700, 720, 580, 760, 690, 640, 750, 710, 660],
        "gender": [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        "employment_status": [0, 2, 0, 3, 0, 0, 1, 2, 0, 1],
        "loan_approved": [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    })

    # Split into reference and current datasets
    reference = data.sample(frac=0.6, random_state=42)
    current = data.drop(reference.index)

    X_ref = reference.drop(columns=["loan_approved"])
    X_cur = current.drop(columns=["loan_approved"])

    reference["prediction"] = model.predict(X_ref)
    current["prediction"] = model.predict(X_cur)

    # ---- Data Drift Detection using KS Test ----
    drift_results = []
    for col in X_ref.columns:
        stat, p_val = ks_2samp(reference[col], current[col])
        drift_detected = p_val < 0.05
        drift_results.append((col, p_val, drift_detected))

    print("\n Data Drift Results (KS Test):")
    for col, p_val, drift in drift_results:
        status = "DRIFT" if drift else "NO DRIFT"
        print(f"  - {col}: {status} (p={p_val:.4f})")

    # ---- Fairness Evaluation ----
    fairness_output = ""
    if AIF360_AVAILABLE:
        def make_aif_dataset(df):
            return BinaryLabelDataset(
                df=df,
                label_names=["prediction"],
                protected_attribute_names=["gender"]
            )

        aif_ref = make_aif_dataset(reference)
        metric = BinaryLabelDatasetMetric(
            aif_ref,
            privileged_groups=[{'gender': 1}],
            unprivileged_groups=[{'gender': 0}]
        )
        fairness_output += "\n   Fairness Metrics (AIF360):\n"
        fairness_output += f"  - Disparate Impact: {metric.disparate_impact():.4f}\n"
        fairness_output += f"  - Mean Difference: {metric.mean_difference():.4f}\n"
        fairness_output += f"  - Statistical Parity: {metric.statistical_parity_difference():.4f}\n"
    else:
        # Manual fallback if AIF360 not installed
        male_preds = reference[reference["gender"] == 1]["prediction"]
        female_preds = reference[reference["gender"] == 0]["prediction"]
        mean_diff = male_preds.mean() - female_preds.mean()
        fairness_output += "\n Fairness Metrics (Manual):\n"
        fairness_output += f"  - Mean Approval Difference (M-F): {mean_diff:.4f}\n"

    print(fairness_output)

    # ---- Save Results to File ----
    report_path = os.path.join(REPORTS_DIR, "monitoring_results.txt")
    with open(report_path, "w") as f:
        f.write("Data Drift Results (KS Test):\n")
        for col, p_val, drift in drift_results:
            f.write(f"{col}: {'DRIFT' if drift else 'NO DRIFT'} (p={p_val:.4f})\n")
        f.write(fairness_output)

    print(f"\n Monitoring report saved to: {report_path}")


if __name__ == "__main__":
    run_monitoring()'''
    
'''  
import pandas as pd
import joblib
import os
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# Optional fairness fallback
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False

def plot_drift(drift_results, save_path):
    features = [x[0] for x in drift_results]
    p_values = [x[1] for x in drift_results]
    drift_flags = [x[2] for x in drift_results]

    colors = ['red' if drift else 'green' for drift in drift_flags]
    
    plt.figure(figsize=(8,4))
    plt.bar(features, p_values, color=colors)
    plt.axhline(y=0.05, color='blue', linestyle='--', label='Drift threshold (p=0.05)')
    plt.ylabel('p-value (KS Test)')
    plt.title('Feature Data Drift Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_monitoring():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)

    # Sample data
    data = pd.DataFrame({
        "income": [35000, 42000, 52000, 29000, 61000, 48000, 33000, 58000, 44000, 39000],
        "credit_score": [650, 700, 720, 580, 760, 690, 640, 750, 710, 660],
        "gender": [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        "employment_status": [0, 2, 0, 3, 0, 0, 1, 2, 0, 1],
        "loan_approved": [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    })

    # Split data
    reference = data.sample(frac=0.6, random_state=42)
    current = data.drop(reference.index)

    X_ref = reference.drop(columns=["loan_approved"])
    X_cur = current.drop(columns=["loan_approved"])

    reference["prediction"] = model.predict(X_ref)
    current["prediction"] = model.predict(X_cur)

    # Data Drift: KS Test
    drift_results = []
    for col in X_ref.columns:
        stat, p_val = ks_2samp(reference[col], current[col])
        drift_detected = p_val < 0.05
        drift_results.append((col, p_val, drift_detected))

    print("\nData Drift Results (KS Test):")
    for col, p_val, drift in drift_results:
        status = "DRIFT" if drift else "NO DRIFT"
        print(f"  - {col}: {status} (p={p_val:.4f})")

    # Fairness Metrics
    fairness_output = ""
    if AIF360_AVAILABLE:
        def make_aif_dataset(df):
            return BinaryLabelDataset(
                df=df,
                label_names=["prediction"],
                protected_attribute_names=["gender"]
            )

        aif_ref = make_aif_dataset(reference)
        metric = BinaryLabelDatasetMetric(aif_ref,
                                          privileged_groups=[{'gender': 1}],
                                          unprivileged_groups=[{'gender': 0}])
        fairness_output += "<h3>Fairness Metrics (AIF360):</h3>"
        fairness_output += f"<p>Disparate Impact: {metric.disparate_impact():.4f}</p>"
        fairness_output += f"<p>Mean Difference: {metric.mean_difference():.4f}</p>"
        fairness_output += f"<p>Statistical Parity: {metric.statistical_parity_difference():.4f}</p>"
    else:
        # Manual fairness approximation
        male_preds = reference[reference["gender"] == 1]["prediction"]
        female_preds = reference[reference["gender"] == 0]["prediction"]
        mean_diff = male_preds.mean() - female_preds.mean()
        fairness_output += "<h3>Fairness Metrics:</h3>"
        fairness_output += f"<p>Mean Approval Difference (M-F): {mean_diff:.4f}</p>"

    print(fairness_output)

    # Plot drift results
    plot_path = os.path.join(REPORTS_DIR, "data_drift_plot.png")
    plot_drift(drift_results, plot_path)

    # Write HTML report
    html_report_path = os.path.join(REPORTS_DIR, "monitoring_report.html")
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Monitoring Report</title></head><body>")
        f.write("<h2>Data Drift Results </h2><ul>")
        for col, p_val, drift in drift_results:
            status = "DRIFT" if drift else "NO DRIFT"
            f.write(f"<li>{col}: {status} (p={p_val:.4f})</li>")
        f.write("</ul>")
        f.write(fairness_output)
        f.write(f'<h2>Drift Plot</h2><img src="data_drift_plot.png" alt="Data Drift Plot">')
        f.write("</body></html>")

    print(f"\nMonitoring report saved to: {html_report_path}")

if __name__ == "__main__":
    run_monitoring()'''
import pandas as pd
import joblib
import os
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# Try importing AIF360
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False

def plot_drift(drift_results, save_path):
    features = [x[0] for x in drift_results]
    p_values = [x[1] for x in drift_results]
    drift_flags = [x[2] for x in drift_results]
    colors = ['red' if drift else 'green' for drift in drift_flags]

    plt.figure(figsize=(8, 4))
    plt.bar(features, p_values, color=colors)
    plt.axhline(y=0.05, color='blue', linestyle='--', label='Drift threshold (p=0.05)')
    plt.ylabel('p-value (KS Test)')
    plt.title('Feature Data Drift Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_monitoring():
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Example input data
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

    # ---- Drift Detection ----
    drift_results = []
    for col in X_ref.columns:
        stat, p_val = ks_2samp(reference[col], current[col])
        drift_detected = p_val < 0.05
        drift_results.append((col, p_val, drift_detected))

    # ---- Fairness Metrics ----
    if AIF360_AVAILABLE:
        def make_aif_dataset(df):
            return BinaryLabelDataset(
                df=df,
                label_names=["prediction"],
                protected_attribute_names=["gender"]
            )
        aif_ref = make_aif_dataset(reference)
        metric = BinaryLabelDatasetMetric(aif_ref,
                                          privileged_groups=[{'gender': 1}],
                                          unprivileged_groups=[{'gender': 0}])
        fairness_metrics = {
            "Disparate Impact": metric.disparate_impact(),
            "Mean Difference": metric.mean_difference(),
            "Statistical Parity": metric.statistical_parity_difference()
        }
    else:
        male_preds = reference[reference["gender"] == 1]["prediction"]
        female_preds = reference[reference["gender"] == 0]["prediction"]
        fairness_metrics = {
            "Mean Approval Difference (M-F)": male_preds.mean() - female_preds.mean()
        }

    # ---- Save Text Report ----
    txt_report_path = os.path.join(REPORTS_DIR, "monitoring_results.txt")
    with open(txt_report_path, "w") as f:
        f.write("Data Drift Results (KS Test):\n")
        for col, p_val, drift in drift_results:
            status = "DRIFT" if drift else "NO DRIFT"
            f.write(f"{col}: {status} (p={p_val:.4f})\n")
        f.write("\nFairness Metrics:\n")
        for key, value in fairness_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"✅ Text report saved to: {txt_report_path}")

    # ---- Save Drift Plot ----
    plot_path = os.path.join(REPORTS_DIR, "data_drift_plot.png")
    plot_drift(drift_results, plot_path)

    # ---- Save HTML Report ----
    html_path = os.path.join(REPORTS_DIR, "monitoring_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Monitoring Report</title></head><body>")
        f.write("<h2>Data Drift Results (KS Test)</h2><ul>")
        for col, p_val, drift in drift_results:
            status = "DRIFT" if drift else "NO DRIFT"
            f.write(f"<li>{col}: {status} (p={p_val:.4f})</li>")
        f.write("</ul>")

        f.write("<h2>Fairness Metrics</h2><ul>")
        for key, value in fairness_metrics.items():
            f.write(f"<li>{key}: {value:.4f}</li>")
        f.write("</ul>")

        f.write(f'<h2>Drift Plot</h2><img src="data_drift_plot.png" alt="Data Drift Plot">')
        f.write("</body></html>")

    print(f"✅ HTML report saved to: {html_path}")
    print(f"✅ Drift plot saved to: {plot_path}")

if __name__ == "__main__":
    run_monitoring()

