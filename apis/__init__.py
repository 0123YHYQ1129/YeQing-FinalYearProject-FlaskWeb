"""Flask based backend APIs"""
from flask import Blueprint
from flask_restx import Api

# Define API responses
api_response = {
    200: "OK",
    201: "Created",
    400: "Invalid Inputs",
    404: "Not Found",
    500: "Other Error",
}

# Create a blueprint for the API
api_v1 = Blueprint(
    "Advanced Machine Learning API for Predictive Modelling of Traffic Flow",
    __name__,
)

# Create an API instance
api = Api(
    api_v1,
    title="Advanced Machine Learning API for Predictive Modelling of Traffic Flow",
    version="0.4.6",
    description="PyPARK Python Packages & Calculations as a Service",
)

# Import the prediction namespace
from .pred import pred

# Add the prediction namespace to the API
api.add_namespace(pred)

