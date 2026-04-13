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

all_indices=[]

all_bourse=[]

START_DATE = "2021-01-01"

all_dates = pd.date_range(
    start=START_DATE,
    end=pd.Timestamp.today(),
    freq="D"
)


print("📂 Loading all data...\n")

def clean_indice_dataframe(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    
    # ── Reindex sur calendrier global ─────────────────────────────────
    df = df.reindex(all_dates)
    
    # ── Flag jour de trading (AVANT fill) ─────────────────────────────
    df["is_trading_day"] = (~df["Close"].isna()).astype(int)
    
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
    df = df.iloc[54:]
    
    # ── Reset index pour concat ───────────────────────────────────────
    df = df.reset_index().rename(columns={"index": "Date"})
    
    df[indicator_cols] = df[indicator_cols].bfill()
    
    return df

def load_indices(base_path):

    for file in os.listdir(base_path):
        path = os.path.join(base_path, file)
        
        if not file.endswith(".csv"):
            continue
        
        df = pd.read_csv(path)
        
        df = clean_indice_dataframe(df)

        df["indice"] = file.replace(".csv", "")
        
        all_indices.append(df)

    return all_indices

def load_bourse(base_path):
    for file in os.listdir(base_path):
        path = os.path.join(base_path, file)
        if not file.endswith(".csv"):
            continue
        
        df = pd.read_csv(path)
        df = clean_indice_dataframe(df)

        df["indice"] = file.replace(".csv", "")

        all_bourse.append(df)

    return all_bourse

def load_actions(base_path):
    all_data = []

    for market in os.listdir(base_path):
        market_path = os.path.join(base_path, market)

        if not os.path.isdir(market_path):
            continue

        for sector in os.listdir(market_path):
            sector_path = os.path.join(market_path, sector)

            if not os.path.isdir(sector_path):
                continue

            for file in os.listdir(sector_path):

                if not file.endswith(".csv"):
                    continue

                file_path = os.path.join(sector_path, file)

                try:
                    df = pd.read_csv(file_path)
                    
                    df = clean_action_dataframe(
                        df,
                        market=market,
                        sector=sector,
                        ticker=file.replace(".csv", "")
                    )
                    df["Ticker"] = file.replace(".csv", "")
                    cols_to_check = ["Open", "High"]
                    if df[cols_to_check].isna().any().any(): 
                        continue 
                    all_data.append(df)

                except Exception as e:
                    print(f"❌ Error {file}: {e}")

    return all_data

def clean_action_dataframe(df,market,sector,ticker):
    
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
    
    df[indicator_cols] = df[indicator_cols].bfill()
    
   
    return df



# ─── MAIN ────────────────────────────────────────────────────────────────────
def dataframe_info(df):
    print(df.head())
    print(df.info())
    print(df.select_dtypes(include=['object','int','float','bool']).isnull().sum())

def label_encoder(df,colonne,labelencoder):
    df[colonne]= labelencoder.fit_transform(df[colonne])

def clean_all_dataframe(fill_df,fill_indice,fill_bourse):
    fill_df = pd.get_dummies(fill_df, columns=["Market", "Sector"])
    
    labelencoder = LabelEncoder()
    label_encoder(fill_df,'Ticker',labelencoder)
    label_encoder(fill_indice,'indice',labelencoder)
    label_encoder(fill_bourse,'indice',labelencoder)

def fusion(liste):
    df=pd.concat(liste, ignore_index=True)
    return df

def load_all():
    indices = load_indices("data_base/sector_indices")
    bourses=load_bourse("data_base/bourse_indices")
    # actions
    actions = load_actions("data_base")
    print(f"\n✅ Loaded {len(actions)+len(indices)+len(bourses)} files")
    return indices,bourses,actions

def concat_dataframe():
    
    indices,bourses,actions=load_all()
    
    # ── Fusion ──
    fill_indice= fusion(indices)
    fill_action = fusion(actions)
    fill_bourse = fusion(bourses)
    return fill_indice,fill_action,fill_bourse
    

def main():  
    
    fill_indice,fill_action,fill_bourse = concat_dataframe()
    
    clean_all_dataframe(fill_action,fill_indice,fill_bourse)
    
    dataframe_info(fill_action)
    dataframe_info(fill_bourse)
    dataframe_info(fill_indice)
    return fill_indice,fill_action,fill_bourse

if __name__ == "__main__":
    fill_indice,fill_action,fill_bourse=main()









