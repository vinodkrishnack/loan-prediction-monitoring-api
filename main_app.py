'''import sys
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# ✅ Define your FastAPI app FIRST
main_app = FastAPI(title="Combined Loan Prediction & Monitoring API")

# ✅ Now you can safely use main_app
@main_app.get("/")
async def root():
    return {
        "message": "Welcome to the Combined API. Visit /predict/ or /monitoring/."
    }
    # Or auto-redirect:
    # return RedirectResponse(url="/predict")

# Add your app folders to sys.path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))
sys.path.append(os.path.join(os.path.dirname(__file__), "app1"))

# Import the FastAPI app instances from your existing apps
from app.main import app as prediction_app
from app1.main import app as monitoring_app

# Mount each app at a prefix path
main_app.mount("/predict", prediction_app)
main_app.mount("/monitoring", monitoring_app)'''

import sys
import os
from fastapi import FastAPI

# Add app folders to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))
sys.path.append(os.path.join(os.path.dirname(__file__), "app1"))

# Import individual FastAPI apps
from app.main import app as prediction_app
from app1.main import app as monitoring_app

# Create main combined app
main_app = FastAPI(title="Combined Loan Prediction & Monitoring API")

# Root endpoint
@main_app.get("/")
async def root():
    return {
        "message": "Welcome to the Combined API. Visit /predict/ or /monitoring/."
    }

# Mount the apps
main_app.mount("/predict", prediction_app)
main_app.mount("/monitoring", monitoring_app)
