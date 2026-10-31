import os
import numpy as np
import pandas as pd
import yfinance as yf
from src.models import Stock

ASSUMED_BASIS = "Assumed target (simulated)"
HISTORICAL_BASIS = "Trailing 1Y return (historical)"
DEFAULT_VOL = 0.25

def ensure_preset_datasets_exist():
    """Auto-generates synthetic preset benchmarks if missing."""
    os.makedirs("data", exist_ok=True)
    for n in [10, 15, 20, 25, 30]:
        path = f"data/stocks_n{n}.csv"
        if not os.path.exists(path):
            np.random.seed(42 + n)
            sample_stocks = []
            for i in range(1, n + 1):
                cost = float(np.random.randint(200, 5000))
                return_val = float(cost * np.random.uniform(1.05, 1.35))
                sample_stocks.append({
                    "id": i,
                    "name": f"Asset_{i}",
                    "cost": round(cost, 2),
                    "return_val": round(return_val, 2)
                })
            pd.DataFrame(sample_stocks).to_csv(path, index=False)

def fetch_live_nse_data(return_basis: str) -> tuple[list[Stock], list[str]]:
    """Fetches real-time batch prices and 1Y historical volatility for an expanded master list of NSE stocks."""
    
    tickers_map = {
        "HDFCBANK.NS": ("HDFC Bank", 1.18, "Banking"),
        "ICICIBANK.NS": ("ICICI Bank", 1.17, "Banking"),
        "SBIN.NS": ("State Bank of India", 1.17, "Banking"),
        "AXISBANK.NS": ("Axis Bank", 1.18, "Banking"),
        "KOTAKBANK.NS": ("Kotak Mahindra Bank", 1.16, "Banking"),
        "BAJFINANCE.NS": ("Bajaj Finance", 1.20, "Banking"),
        "TCS.NS": ("Tata Consultancy Services", 1.16, "IT"),
        "INFY.NS": ("Infosys", 1.19, "IT"),
        "WIPRO.NS": ("Wipro", 1.14, "IT"),
        "HCLTECH.NS": ("HCL Technologies", 1.17, "IT"),
        "TECHM.NS": ("Tech Mahindra", 1.15, "IT"),
        "RELIANCE.NS": ("Reliance Industries", 1.18, "Energy"),
        "NTPC.NS": ("NTPC Ltd", 1.15, "Energy"),
        "POWERGRID.NS": ("Power Grid Corp", 1.13, "Energy"),
        "ONGC.NS": ("ONGC", 1.14, "Energy"),
        "ITC.NS": ("ITC Ltd", 1.14, "FMCG"),
        "HINDUNILVR.NS": ("Hindustan Unilever", 1.12, "FMCG"),
        "NESTLEIND.NS": ("Nestle India", 1.13, "FMCG"),
        "BRITANNIA.NS": ("Britannia Industries", 1.14, "FMCG"),
        "MARUTI.NS": ("Maruti Suzuki", 1.18, "Auto"),
        "M&M.NS": ("Mahindra & Mahindra", 1.19, "Auto"),
        "LT.NS": ("Larsen & Toubro", 1.18, "Infrastructure"),
        "ULTRACEMCO.NS": ("UltraTech Cement", 1.16, "Infrastructure"),
        "SUNPHARMA.NS": ("Sun Pharma", 1.19, "Healthcare"),
        "DRREDDY.NS": ("Dr. Reddy's Labs", 1.17, "Healthcare"),
        "CIPLA.NS": ("Cipla", 1.16, "Healthcare"),
        "TATASTEEL.NS": ("Tata Steel", 1.15, "Other"),
        "JSWSTEEL.NS": ("JSW Steel", 1.15, "Other"),
        "GRASIM.NS": ("Grasim Industries", 1.14, "Other"),
        "BHARTIARTL.NS": ("Bharti Airtel", 1.18, "Telecom"),
        "TITAN.NS": ("Titan Company", 1.17, "Other"),
        "NIFTYBEES.NS": ("Nifty 50 ETF", 1.15, "ETF"),
        "GOLDBEES.NS": ("Gold BeES ETF", 1.12, "ETF"),
        "BANKBEES.NS": ("Nifty Bank ETF", 1.16, "ETF")
    }

    stocks_list: list[Stock] = []
    notes: list[str] = []
    missing: list[str] = []

    try:
        data = yf.download(list(tickers_map.keys()), period="1y", progress=False)['Close']
        idx = 1
        for symbol, (name, assumed_mult, sector) in tickers_map.items():
            series = data[symbol].dropna() if symbol in data and symbol in data.columns else None
            if series is None or series.empty:
                missing.append(symbol)
                continue

            price = float(series.iloc[-1])
            daily = series.pct_change().dropna()
            if len(daily) >= 30:
                volatility = float(daily.std() * np.sqrt(252))
                trailing = price / float(series.iloc[0]) - 1.0
            else:
                volatility, trailing = DEFAULT_VOL, None

            multiplier = 1.0 + trailing if return_basis == HISTORICAL_BASIS and trailing is not None else assumed_mult

            stocks_list.append(
                Stock(
                    id=idx,
                    name=f"{name} ({symbol.replace('.NS', '')})",
                    cost=round(price, 2),
                    return_val=round(price * multiplier, 2),
                    volatility=round(volatility, 4),
                    sector=sector,
                )
            )
            idx += 1

        if not stocks_list:
            raise ValueError("No price data returned")
        if missing:
            notes.append(f"Skipped {len(missing)} inactive tickers.")
    except Exception as e:
        notes.append(f"Unable to reach live feeds: {e}. Showing offline quotes.")
        fallback_data = [
            ("Reliance Industries", 2950.0, 3480.0, 0.22, "Energy"),
            ("TCS", 4120.0, 4780.0, 0.22, "IT"),
            ("HDFC Bank", 1680.0, 1980.0, 0.20, "Banking"),
            ("Infosys", 1850.0, 2200.0, 0.25, "IT"),
            ("Nifty 50 ETF", 270.0, 310.0, 0.15, "ETF"),
            ("Gold BeES ETF", 68.0, 78.0, 0.12, "ETF")
        ]
        stocks_list = [
            Stock(id=i+1, name=item[0], cost=item[1], return_val=item[2], volatility=item[3], sector=item[4])
            for i, item in enumerate(fallback_data)
        ]

    return stocks_list, notes
