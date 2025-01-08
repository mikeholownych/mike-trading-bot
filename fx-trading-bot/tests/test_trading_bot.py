
import pytest
from bot.trading_bot import lambda_handler

def test_lambda_handler_buy():
    event = {"symbol": "EURUSD", "timeframe": "M1"}
    response = lambda_handler(event, None)
    assert response["statusCode"] == 200
    assert "body" in response

def test_lambda_handler_no_trade():
    event = {"symbol": "EURUSD", "timeframe": "M1"}
    response = lambda_handler(event, None)
    assert response["statusCode"] == 200
    assert response["body"] == '{"status": "no trade"}'
            