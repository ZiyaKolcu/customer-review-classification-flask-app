import pytest
from app import app


# This fixture creates a test client for the Flask app
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


# Basic test to check if home page is reachable
def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Customer Review Classification" in response.data


# Test the prediction route with valid input and model
def test_predict_valid(client):
    response = client.post(
        "/predict",
        data={"input_value": "This product is great!", "model_name": "model-1"},
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert "prediction" in json_data
    assert json_data["prediction"] in ["positive", "negative"]


# Test the prediction route with invalid model
def test_predict_invalid_model(client):
    response = client.post(
        "/predict", data={"input_value": "Bad quality!", "model_name": "invalid-model"}
    )
    assert response.status_code == 400
