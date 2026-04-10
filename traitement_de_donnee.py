# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:40:17 2026

@author: grabe
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

BASE_PATH = "data_base"

all_data = []

all_indice=[]

all_bourse=[]

START_DATE = "2021-01-01"

all_dates = pd.date_range(
    start=START_DATE,
    end=pd.Timestamp.today(),
    freq="D"
)
print("📂 Loading all data...\n")

for market in os.listdir(BASE_PATH):
    market_path = os.path.join(BASE_PATH, market)
    
    if not os.path.isdir(market_path):
        continue

    for sector in os.listdir(market_path):
        sector_path = os.path.join(market_path, sector)
        
        if not os.path.isdir(sector_path):
            df = pd.read_csv(sector_path)
            
            
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            
            # ── Reindex sur calendrier global ─────────────────────────────────
            df = df.reindex(all_dates)
            
            # ── Flag jour de trading (AVANT fill) ─────────────────────────────
            df["is_trading_day"] = (~df["Close"].isna()).astype(int)
            
            df["indice"] = sector.replace(".csv", "")
            price_cols = ["Open", "High", "Low", "Close"]
            indicator_cols = [
                "MA20", "MA50", "Momentum", "RSI",
                "MACD", "MACD_Signal", "MACD_Hist",
                "Volatility_20d",
                "BB_Upper", "BB_Lower", "BB_Mid","Return_1d","Return_5d",
                "Normalized"
            ]
            # Prix → forward fill
            df[price_cols] = df[price_cols].ffill()
            
            
            # Volume → 0 si marché fermé
            df['Volume'] = df['Volume'].fillna(0)
            
            # Indicateurs → forward fill
            existing_indicators = [col for col in indicator_cols if col in df.columns]
            df[existing_indicators] = df[existing_indicators].ffill()
            
            # ── Supprimer début série (indicateurs incomplets) ────────────────
            df = df.iloc[4:]
            
            # ── Reset index pour concat ───────────────────────────────────────
            df = df.reset_index().rename(columns={"index": "Date"})
            
            df[indicator_cols] = df[indicator_cols].bfill()
            
            if df['indice'][0]== 'cac40' or df['indice'][0]== 'sti' or df['indice'][0]== 'sp500': 
                all_bourse.append(df)
            else:
                all_indice.append(df)
            continue





        for file in os.listdir(sector_path):
            
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(sector_path, file)

            try:
                df = pd.read_csv(file_path)
                # ── Préparation dates ─────────────────────────────────────────────
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
                
                # ── Reindex sur calendrier global ─────────────────────────────────
                df = df.reindex(all_dates)
                
                # ── Flag jour de trading (AVANT fill) ─────────────────────────────
                df["is_trading_day"] = (~df["Close"].isna()).astype(int)
                
                # ── Remettre metadata ─────────────────────────────────────────────
                df["Market"] = market
                df["Sector"] = sector
                df["Ticker"] = file.replace(".csv", "")
                
                # ── Colonnes ──────────────────────────────────────────────────────
                price_cols = ["Open", "High", "Low", "Close"]
                volume_cols = [col for col in df.columns if "Volume" in col]
                
                indicator_cols = [
                    "MA20", "MA50", "Momentum", "RSI",
                    "MACD", "MACD_Signal", "MACD_Hist",
                    "Volatility_20d",
                    "BB_Upper", "BB_Lower", "BB_Mid","VIX_Close"
                ]
                
                # ── Fill intelligent ──────────────────────────────────────────────
                
                # Prix → forward fill
                df[price_cols] = df[price_cols].ffill()
                
                
                # Volume → 0 si marché fermé
                df[volume_cols] = df[volume_cols].fillna(0)
                
                # Indicateurs → forward fill
                existing_indicators = [col for col in indicator_cols if col in df.columns]
                df[existing_indicators] = df[existing_indicators].ffill()
                
                # ── Supprimer début série (indicateurs incomplets) ────────────────
                df = df.iloc[4:]
                
                # ── Reset index pour concat ───────────────────────────────────────
                df = df.reset_index().rename(columns={"index": "Date"})
                
                
                cols_to_check = ["Open", "High"]
                if df[cols_to_check].isna().any().any(): 
                    continue 
                

                df[indicator_cols] = df[indicator_cols].bfill()
                
                # ── Ajouter au dataset global ─────────────────────────────────────
                all_data.append(df)
                
                
                
                

            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

print(f"\n✅ Loaded {len(all_data)+len(all_indice)} files")

# ── Fusion ──
fill_indice= pd.concat(all_indice, ignore_index=True)
fill_df = pd.concat(all_data, ignore_index=True)
fill_bourse = pd.concat(all_bourse, ignore_index=True)

fill_df = pd.get_dummies(fill_df, columns=["Market", "Sector"])




labelencoder = LabelEncoder()
fill_df['Ticker']= labelencoder.fit_transform(fill_df['Ticker'])
fill_indice['indice']=labelencoder.fit_transform(fill_indice['indice'])

fill_bourse['indice']=labelencoder.fit_transform(fill_bourse['indice'])

deletelement=['CAC40_Close','CAC40_High','CAC40_Low','CAC40_Open','CAC40_Volume',
              'SP500_Close','SP500_High','SP500_Low','SP500_Open','SP500_Volume',
              'STI_Close','STI_High','STI_Low','STI_Open','STI_Volume']
for i in deletelement:
    fill_df = fill_df.drop(i,axis=1)

print(fill_df.filter(like="Market_").head())
print(fill_df.filter(like="Sector_").head())
print("\n📊 Final dataset shape :", fill_df.shape)
print(fill_df.head())
print(fill_df.info())
print(fill_df.select_dtypes(include=['object','int','float','bool']).isnull().sum())
print(fill_indice.select_dtypes(include=['object','int','float','bool']).isnull().sum())
print(fill_bourse.select_dtypes(include=['object','int','float','bool']).isnull().sum())










