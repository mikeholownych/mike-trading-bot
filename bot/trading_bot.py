
import json
import stripe
from bot.strategy import generate_signal
from bot.utils import fetch_data, execute_trade
import logging
import boto3
from datetime import datetime

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("SimulatedTrades")

def record_trade_decision(signal, symbol, price):
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    table.put_item(
        Item={
            "Timestamp": timestamp,
            "Signal": signal,
            "Symbol": symbol,
            "Price": price
        }
    )
    print(f"Trade recorded: {signal} {symbol} at ${price}")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def make_trade_decision(signal, symbol, price):
    logger.info(f"Trade decision made: {signal} for {symbol} at ${price}")
    # Simulated trade execution logic

# Stripe API Key
stripe.api_key = "sk_test_your_api_key"

def lambda_handler(event, context):
    action = event.get("action", "trade")
    
    if action == "fund":
        amount = event.get("amount")
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": "Fund Trading Account"},
                    "unit_amount": int(amount * 100),
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url="https://your-website.com/success",
            cancel_url="https://your-website.com/cancel",
        )
        return {"statusCode": 200, "body": json.dumps({"url": session.url})}

    # Main trading logic
    symbol = event.get("symbol", "EURUSD")
    timeframe = event.get("timeframe", "M1")

    data = fetch_data(symbol, timeframe, 100)
    signals = generate_signal(data)
    latest_signal = signals.iloc[-1]['Signal']

    if latest_signal == 1:
        result = execute_trade(symbol, "buy")
        record_trade(signal, symbol, price)
    elif latest_signal == -1:
        result = execute_trade(symbol, "sell")
        record_trade(signal, symbol, price)
    else:
        result = {"status": "no trade"}

    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }
