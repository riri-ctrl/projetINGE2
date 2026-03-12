# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:34:19 2026

@author: grabe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:56:33 2026

@author: grabe
"""

from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf


tickers=['EN.PA', 'OR.PA', 'AI.PA', 'SU.PA', 'AC.PA','VIE.PA','MC.PA','DG.PA','CAP.PA','GLE.PA',
       ]

# erreur avec 'ATO.PA'

def hier():
    return datetime.now(timezone.utc) - timedelta(days=1)

dates = hier()

def load (ticker):
    data = yf.download(ticker, start="2021-01-01", end=dates.date())
    data.columns = data.columns.droplevel(1)
    data = data.reset_index()      # transforme la date en colonne
    data["ticker"] = ticker        # ajoute la colonne ticker
    data["monnaie"]= yf.Ticker(ticker).info["currency"]
    data["secteur"]= yf.Ticker(ticker).info["sector"]
    data["industrie"]= yf.Ticker(ticker).info["industry"]
    data["marketCap"]= yf.Ticker(ticker).info["marketCap"]
    data["P/E ratio"]= yf.Ticker(ticker).info["trailingPE"]  #ratio Cours/Bénéfice
    data["nombre d'action"]= yf.Ticker(ticker).info["sharesOutstanding"]
    print(ticker)
    return data
    


df=pd.concat(list(map(load,tickers)))
df.to_csv("stocks_dataset2.csv", index=False)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    