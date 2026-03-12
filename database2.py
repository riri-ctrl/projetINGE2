# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:56:33 2026

@author: grabe
"""

from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf


tickers=['EN.PA', 'OR.PA', 'AI.PA', 'SU.PA', 'AC.PA','VIE.PA','MC.PA','DG.PA','ATO.PA','CAP.PA','GLE.PA',
       'LR.PA','SAN.PA','BNP.PA','ENG.PA','SW.PA','VIV.PA','CA.PA','BN.PA','SGO.PA','ORA.PA','ML.PA',
       'AIR.PA','RI.PA','KER.PA','WLN.PA','ACA.PA','HO.PA','COFA.PA', 'ALSGD.PA'
       'ALSRS.PA', 'ALNEV.PA', 'ALIBR.PA', 'ALNRG.PA', 'CO.PA', 'ALHPI.PA', 'ALTD.PA', 'ALTAO.PA',
        'ALSEN.PA', 'ALHAF.PA', 'NACON.PA', 'TTE.PA', 'STLAP.PA', 'ALCBX.PA', 'ALEUP.PA', 'SIGHT.PA',
        'ALMKT.PA', 'CS.PA', 'ALTHO.PA', 'DSY.PA', 'STMPA.PA', 'AF.PA', 'ETL.PA', 'ALCPB.PA', 'LAT.PA',
        'ALO.PA', 'RNO.PA', 'DBV.PA', 'ALPWG.PA', 'BOL.PA', 'ALCBI.PA', 'UBI.PA', 'ALHG.PA', 'VLA.PA', 'EDEN.PA',
        'FR.PA', 'BVI.PA', 'FRVIA.PA', 'SPIE.PA', 'AYV.PA', 'ALDBT.PA', 'LI.PA', 'LR.PA', 'PUB.PA', 'RXL.PA',
        'SAF.PA', 'SESG.PA', 'ELIOR.PA', 'ALAGO.PA', 'EL.PA', 'VK.PA', 'SCR.PA', 'FDJU.PA', 'GET.PA', 'ALGID.PA',
        'AVT.PA', 'EAPI.PA', 'CRUDP.PA', 'AL2SI.PA', 'TEP.PA', 'URW.PA', 'ALROC.PA', 'ELIS.PA', 'TE.PA', 'ALALO.PA',
        'ATO.PA', 'CLARI.PA', 'ENX.PA', 'TFI.PA', 'FGR.PA', 'MLGEQ.PA', 'ALCOX.PA', 'MERY.PA', 'AKE.PA', 'ERF.PA',
        'NXI.PA', 'GNFT.PA', 'ALTME.PA', 'ALKAL.PA', 'RUI.PA', 'NANO.PA', 'ALCLS.PA', 'PLX.PA', 'DEC.PA', 'MEDCL.PA',
        'HRL', 'CSGP', 'FTV', 'HST', 'TSLA', 'AIG', 'DBV', 'WRB', 'LOW', 'CSX', 'BDX', 'UBER', 'KMB',
         'ARES', 'INTU', 'HOOD', 'J', 'PLD', 'ULTA', 'COP', 'DHI', 'APO', 'MTCH', 'PKG', 'CTAS', 'SBAC',
         'TKO', 'INCY',  'AMGN', 'EG', 'REG', 'ICE', 'VTR', 'HUBB', 'HON', 'MU', 'CINF', 'NUE', 'FE',
         'AMAT', 'TROW', 'INVH', 'RSG', 'AJG', 'ABBV', 'T', 'JBHT', 'HPE',  'PNR', 'AVGO', 
         'SYF', 'IFF', 'XOM', 'DELL', 'PFE',  'TFC', 'IBM', 'WEC',  'MTB', 'WYNN', 'HOLX',
         'KDP', 'OMC', 'AKAM', 'HLT', 'TT', 'BSX', 'YUM', 'RVTY', 'STLD', 'DD', 'ALB',
         'NEE', 'COF', 'EME', 'LRCX', 'CCL', 'VMC',  'APA', 'GD', 'NRG', 'HD', 'AMCR', 'LNT', 'FSLR',
         'KKR', 'URI', 'UDR', 'RTX', 'EIX', 'EQR', 'TSN', 'ROP', 'PEG', 'MDT', 'NDAQ', 'BLK', 'TSCO', 'DSY',
          'EOG', 'KEYS', 'WMB', 'DLR', 'HSIC', 'MTD','MET', 'NTRS', 'CRH', 'IDXX', 'PAYC', 'XYZ', 'AVT',
         'GWW', 'DOW', 'KEY', 'IT', 'MCK', 'AXON',  'ORLY', 'CTSH', 'CI', 'ODFL', 'AXP', 'WBD', 'CHD',
          'SAF', 'APD', 'DVA', 'SW', 'LW', 'NCLH', 'AMT', 'SRE', 'EBAY', 'DUK', 'MOH',  'AME',
         'ADM', 'AWK', 'PM', 'TRMB', 'WMT', 'TTE', 'VLO', 'LDOS', 'PSA', 'MAR', 'ED', 'KIM', 'CTRA', 'ROK',
         'GIS', 'FIS', 'CHTR', 'LHX', 'TPL', 'ORCL', 'VZ', 'TJX', 'DXCM', 'BALL',
         'IQV',  'TTWO', 'SPGI', 'MSFT', 'BMY',  'DVN', 'GEN', 'TPR', 'XYL',
         'MKC', 'UNH', 'FISV', 'VRSN',  'BLDR', 'EVRG', 'AMP', 'MDLZ', 'MPWR', 'UAL', 'UNP', 'SCHW',
         'CPRT', 'CIEN',  'LYV',  'AZO', 'GPN', 'EXPD', 'MGM', 'SBUX', 'WTW', 'LI', 'ABT', 'CL',
         'SMCI', 'ECL', 'JCI', 'D', 'RF', 'BK', 'ELV', 'HPQ', 'FCX', 'SHW', 'SJM', 'BBY', 'LII', 'KO',
         'ISRG', 'CNP', 'IEX', 'RXL', 'GOOG', 'LYB', 'MCHP', 'TAP', 'AVB', 'BAC', 'GEV', 'FAST',
         'CDW', 'STZ', 'DIS', 'DOV', 'HAS', 'FR', 'HIG', 'STX', 'DASH', 'TMUS', 'AFL',
         'ADSK', 'LUV', 'WST', 'TRV', 'HAL', 'LULU', 'GM', 'GPC', 'PLTR', 'BR', 'PH', 'COO', 'OKE', 'ALGN',
         'FDX', 'MCD', 'Q', 'KVUE', 'PEP', 'STT', 'SNA', 'CMS', 'WM', 'BA', 'MA', 'XEL',
         'CBRE', 'ZBH', 'CVX', 'BEN', 'CNC', 'AEE', 'UPS', 'CPB', 'INTC', 'DAL', 'STE',
         'ZBRA', 'PLX', 'FICO', 'EQIX', 'ALCBX', 'PAYX', 'TMO', 'AMD', 'QCOM', 'LMT', 'TECH', 'NFLX', 'BKNG',
         'MSCI', 'SOLV', 'COST', 'C', 'WFC', 'CAG', 'LEN', 'TGT', 'PRU', 'LH', 'COR', 'NDSN', 'NVDA', 'GEHC',
         'NWS', 'AON', 'JBL', 'SWKS', 'HBAN', 'SESG', 'TE', 'HUM', 'DEC', 'O', 'WAB', 'ROL', 'TDG', 'AVY', 'EMR',
         'DE', 'VLTO', 'MRK', 'NOW', 'CB', 'BKR', 'NI', 'CEG', 'BRO', 'GRMN', 'SPG', 'ADP', 'PANW',
         'POOL', 'VICI', 'FFIV', 'DGX', 'RMD', 'NTAP', 'META', 'PCAR', 'GLW', 'SYK', 'APH', 'CME', 'CF',
         'ARE', 'CCI', 'EFX', 'CARR', 'OTIS', 'MLM', 'GS', 'RL', 'FANG', 'AMZN', 'MSI', 'RJF',
         'IBKR', 'UHS', 'AES', 'MRNA', 'NWSA', 'CFG', 'MS', 'PTC', 'EL', 'A', 'WSM', 'IP', 'JPM', 'PODD',
         'MO', 'GE', 'AAPL', 'EQT', 'ADI', 'CBOE', 'MOS', 'PPL', 'FOXA', 'ITW', 'NKE',
         'BAX', 'PSKY', 'ERIE', 'DTE', 'PNW', 'MAA', 'WAT', 'DLTR', 'APTV', 'ETN', 'HCA', 'LLY',
         'BG', 'PSX', 'GL', 'ES', 'PCG', 'JKHY', 'SWK', 'TTD', 'ON', 'VTRS', 'ROST', 'AEP', 'TXN',
         'PYPL', 'CDNS', 'ETR', 'FDS', 'FIX', 'NVR', 'ATO', 'GDDY', 'ELIS', 'SYY', 'CHRW', 'ACGL', 'APP',
         'ANET', 'WELL', 'MNST', 'DHR', 'CAH', 'DRI', 'IRM', 'REGN', 'KR', 'EXR', 'LIN', 'KLAC', 'F', 'WY',
         'ALTD', 'IR', 'JNJ', 'DOC', 'RCL', 'CTVA', 'DPZ', 'MAS', 'AOS', 'GILD', 'CVS', 'KHC', 'V',
          'EXPE', 'TFI', 'CLX', 'FRT', 'EA', 'EPAM', 'BIIB', 'BXP', 'CVNA', 'ACN',
         'L', 'TXT', 'BX', 'SNDK', 'GNRC', 'TER', 'PNC', 'USB', 'ALL', 'VRTX', 'MRSH', 'NOC', 'ESS', 'PWR',
         'HSY', 'SNPS', 'LVS', 'PPG', 'HWM', 'TRGP', 'EDEN', 'PGR', 'GOOGL', 'HII', 'KMI', 'CAT',
         'WDAY', 'ABNB', 'FITB', 'CRM', 'ZTS', 'CPAY', 'EW', 'ADBE', 'CPT', 'IVZ', 'CMCSA',
         'DECK', 'AIZ', 'NEM', 'VRSK', 'FTNT', 'ALLE', 'COIN', 'SO', 'MMM', 'VST', 'TYL', 'OXY', 
         'PG', 'CRL', 'TDY', 'NXPI', 'CMI', 'FOX', 'CRWD', 'NSC', 'DG',
         'EXC', 'PHM', 'TEL', 'CMG', 'MPC', 'PFG', 'EXE', 'MCO', 'WDC', 'SLB', 'DDOG']



def hier():
    return datetime.now(timezone.utc) - timedelta(days=1)

dates = hier()

def load (ticker):
    data = yf.download(ticker, start="2021-01-01", end=dates.date())
    data.columns = data.columns.droplevel(1)
    data = data.reset_index()      # transforme la date en colonne
    data["ticker"] = ticker        # ajoute la colonne ticker
    data["monnaie"]= yf.Ticker(ticker).info["currency"]
    return data
    


df=pd.concat(list(map(load,tickers)))
df.to_csv("stocks_dataset.csv", index=False)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    