
# FX Trading Bot

## Overview
This project contains an automated FX trading bot built for deployment on AWS Lambda. It includes funding via Stripe and integrates machine learning strategies for trading.

## Setup Instructions

### 1. Create AWS and Broker Accounts
- Set up an AWS account and configure IAM roles for Lambda.
- Open an MT5 broker account and obtain your login credentials.

### 2. Deploy the Bot
- Install AWS SAM CLI: `pip install aws-sam-cli`
- Deploy the bot: `sam build && sam deploy --guided`

### 3. Configure Stripe
- Sign up for a Stripe account.
- Replace `sk_test_your_api_key` with your Stripe API key in `bot/trading_bot.py`.

### 4. Run Locally
- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest tests/`

## CI/CD
The project includes a GitHub Actions pipeline for continuous integration and deployment.

---
