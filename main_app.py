# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 13:31:56 2025

@author: Admin
"""

import sys
import os
from fastapi import FastAPI

# Add your app folders to sys.path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))
sys.path.append(os.path.join(os.path.dirname(__file__), "app1"))

# Import the FastAPI app instances from your existing apps
from app.main import app as prediction_app
from app1.main import app as monitoring_app

main_app = FastAPI(title="Combined Loan Prediction & Monitoring API")

# Mount each app at a prefix path
main_app.mount("/predict", prediction_app)
main_app.mount("/monitoring", monitoring_app)
