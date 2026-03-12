# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:01:45 2026

@author: grabe
"""

import pandas as pd
import requests

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": "Mozilla/5.0"
}

html = requests.get(url, headers=headers).text
tables = pd.read_html(html)

df = tables[0]

tickers = df["Symbol"].tolist()

print(tickers)


data1=['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APP', 'APTV', 'ACGL', 'ADM', 'ARES', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'XYZ', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CPT', 'CPB', 'COF', 'CAH', 'CCL', 'CARR', 'CVNA', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CIEN', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COIN', 'CL', 'CMCSA', 'FIX', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRH', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG', 'DVA', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EME', 'EMR', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FISV', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'IBKR', 'ICE', 'IFF', 'IP', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MAR', 'MRSH', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PSKY', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'Q', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'HOOD', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SNDK', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TTD', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
data2=['ALSRS', 'ALNEV', 'ALIBR', 'ALNRG', 'CO', 'ALHPI', 'ALTD', 'ALTAO', 'ALSEN', 'ALHAF', 'NACON', 'TTE', 'STLAP', 'ALCBX', 'ALEUP', 'SIGHT', 'ALMKT', 'CS', 'ALTHO', 'DSY', 'STMPA', 'AF', 'ETL', 'ALCPB', 'LAT', 'ALO', 'RNO', 'DBV', 'ALPWG', 'BOL', 'ALCBI', 'UBI', 'ALHG', 'VLA', 'EDEN', 'FR', 'BVI', 'FRVIA', 'SPIE', 'AYV', 'ALDBT', 'LI', 'LR', 'PUB', 'RXL', 'SAF', 'SESG', 'ELIOR', 'ALAGO', 'EL', 'VK', 'SCR', 'FDJU', 'GET', 'ALGID', 'AVT', 'EAPI', 'CRUDP', 'AL2SI', 'TEP', 'URW', 'ALROC', 'ELIS', 'TE', 'ALALO', 'ATO', 'CLARI', 'ENX', 'TFI', 'FGR', 'MLGEQ', 'ALCOX', 'MERY', 'AKE', 'ERF', 'NXI', 'GNFT', 'ALTME', 'ALKAL', 'RUI', 'NANO', 'ALCLS', 'PLX', 'DEC', 'MEDCL', 'COFA', 'ALSGD']

data = list(set(data1 + data2))
print(data)

['HRL', 'CSGP', 'FTV', 'HST', 'TSLA', 'AIG', 'DBV', 'WRB', 'LOW', 'CSX', 'BDX', 'UBER', 'KMB', 'ARES', 'INTU', 'HOOD', 'J', 'PLD', 'ULTA', 'COP', 'DHI', 'APO', 'MTCH', 'PKG', 'ALGID', 'CTAS', 'SBAC', 'TKO', 'INCY', 'BOL', 'AMGN', 'EG', 'REG', 'ICE', 'VTR', 'HUBB', 'HON', 'MU', 'CINF', 'NUE', 'FE', 'AMAT', 'TROW', 'INVH', 'RSG', 'AJG', 'ABBV', 'AF', 'T', 'JBHT', 'HPE', 'CO', 'PNR', 'AVGO', 'RUI', 'SYF', 'IFF', 'XOM', 'DELL', 'PFE', 'ALHAF', 'TFC', 'IBM', 'WEC', 'MEDCL', 'MTB', 'WYNN', 'HOLX', 'KDP', 'OMC', 'AKAM', 'HLT', 'TT', 'BSX', 'YUM', 'RVTY', 'ALSEN', 'ALNEV', 'STLD', 'DD', 'ALB', 'NEE', 'COF', 'EME', 'LRCX', 'CCL', 'VMC', 'ALNRG', 'APA', 'GD', 'NRG', 'HD', 'AMCR', 'LNT', 'FSLR', 'KKR', 'URI', 'UDR', 'RTX', 'EIX', 'EQR', 'TSN', 'ROP', 'PEG', 'MDT', 'NDAQ', 'BLK', 'TSCO', 'DSY', 'AKE', 'EOG', 'KEYS', 'WMB', 'DLR', 'HSIC', 'MTD', 'MET', 'NTRS', 'CRH', 'IDXX', 'PAYC', 'XYZ', 'AVT', 'GWW', 'DOW', 'KEY', 'IT', 'MCK', 'AXON', 'BF.B', 'ORLY', 'CTSH', 'CI', 'ODFL', 'AXP', 'WBD', 'CHD', 'STLAP', 'SAF', 'APD', 'DVA', 'SW', 'LW', 'NCLH', 'AMT', 'SRE', 'EBAY', 'DUK', 'MOH', 'AYV', 'AME', 'ADM', 'AWK', 'PM', 'TRMB', 'WMT', 'TTE', 'VLO', 'LDOS', 'PSA', 'MAR', 'ED', 'KIM', 'CTRA', 'ROK', 'GIS', 'ELIOR', 'ALTHO', 'FIS', 'CHTR', 'GET', 'LHX', 'TPL', 'ORCL', 'VZ', 'TJX', 'GNFT', 'DXCM', 'BALL', 'ALTME', 'IQV', 'STMPA', 'FRVIA', 'TTWO', 'SPGI', 'MSFT', 'BMY', 'MERY', 'DVN', 'GEN', 'TPR', 'XYL', 'MKC', 'UNH', 'FISV', 'VRSN', 'ALDBT', 'BLDR', 'EVRG', 'AMP', 'MDLZ', 'MPWR', 'UAL', 'UNP', 'SCHW', 'CPRT', 'CIEN', 'SCR', 'LYV', 'LR', 'AZO', 'GPN', 'EXPD', 'MGM', 'SBUX', 'WTW', 'LI', 'ABT', 'CL', 'SMCI', 'ECL', 'JCI', 'D', 'RF', 'BK', 'ELV', 'HPQ', 'FCX', 'SHW', 'SJM', 'BBY', 'FGR', 'LII', 'KO', 'NXI', 'ISRG', 'CNP', 'IEX', 'RXL', 'GOOG', 'LYB', 'MCHP', 'TAP', 'AVB', 'BAC', 'GEV', 'FAST', 'BRK.B', 'CDW', 'STZ', 'NACON', 'DIS', 'DOV', 'HAS', 'FR', 'HIG', 'STX', 'CSCO', 'ETL', 'DASH', 'TMUS', 'AFL', 'ADSK', 'LUV', 'ERF', 'WST', 'TRV', 'HAL', 'LULU', 'GM', 'GPC', 'PLTR', 'BR', 'PH', 'COO', 'OKE', 'ALGN', 'FDX', 'AL2SI', 'MCD', 'Q', 'KVUE', 'PEP', 'STT', 'SNA', 'CMS', 'WM', 'BA', 'ALSRS', 'ALO', 'MA', 'XEL', 'CBRE', 'VK', 'ZBH', 'CVX', 'BEN', 'CNC', 'AEE', 'UPS', 'CPB', 'INTC', 'SPIE', 'FDJU', 'DAL', 'STE', 'ZBRA', 'PLX', 'FICO', 'CS', 'EQIX', 'ALCBX', 'PAYX', 'TMO', 'AMD', 'QCOM', 'LMT', 'TECH', 'NFLX', 'BKNG', 'MSCI', 'SOLV', 'COST', 'C', 'WFC', 'CAG', 'LEN', 'TGT', 'PRU', 'LH', 'COR', 'NDSN', 'NVDA', 'GEHC', 'NWS', 'AON', 'JBL', 'SWKS', 'HBAN', 'SESG', 'TE', 'HUM', 'DEC', 'O', 'WAB', 'ROL', 'TDG', 'AVY', 'EMR', 'DE', 'VLTO', 'MRK', 'NOW', 'CB', 'BKR', 'ALKAL', 'NI', 'CEG', 'BRO', 'GRMN', 'SPG', 'ADP', 'PANW', 'POOL', 'VICI', 'FFIV', 'DGX', 'MLGEQ', 'RMD', 'NTAP', 'META', 'PCAR', 'GLW', 'SYK', 'APH', 'CME', 'CF', 'ARE', 'CCI', 'EFX', 'ALEUP', 'CARR', 'OTIS', 'MLM', 'ALTAO', 'GS', 'RL', 'FANG', 'AMZN', 'MSI', 'RJF', 'IBKR', 'UHS', 'AES', 'MRNA', 'NWSA', 'CFG', 'MS', 'PTC', 'EL', 'TEP', 'A', 'WSM', 'IP', 'JPM', 'PODD', 'LAT', 'ALHPI', 'MO', 'GE', 'AAPL', 'EQT', 'ADI', 'CBOE', 'MOS', 'PPL', 'FOXA', 'EAPI', 'ITW', 'NKE', 'SIGHT', 'BAX', 'ALALO', 'PSKY', 'ERIE', 'DTE', 'PNW', 'MAA', 'WAT', 'DLTR', 'APTV', 'ETN', 'HCA', 'LLY', 'BG', 'PSX', 'GL', 'ES', 'PCG', 'JKHY', 'SWK', 'TTD', 'UBI', 'ALHG', 'ON', 'VTRS', 'ROST', 'AEP', 'TXN', 'PYPL', 'CDNS', 'ETR', 'ALCOX', 'FDS', 'FIX', 'NVR', 'ATO', 'GDDY', 'ELIS', 'SYY', 'CHRW', 'ACGL', 'APP', 'ANET', 'WELL', 'MNST', 'DHR', 'CAH', 'ALCPB', 'DRI', 'IRM', 'REGN', 'KR', 'EXR', 'LIN', 'KLAC', 'F', 'WY', 'ALTD', 'ALSGD', 'IR', 'JNJ', 'DOC', 'RCL', 'CTVA', 'DPZ', 'MAS', 'AOS', 'GILD', 'CVS', 'KHC', 'V', 'ALPWG', 'EXPE', 'TFI', 'CLX', 'FRT', 'ALAGO', 'EA', 'ALIBR', 'EPAM', 'BIIB', 'BXP', 'BVI', 'CVNA', 'ACN', 'L', 'TXT', 'BX', 'SNDK', 'GNRC', 'TER', 'PNC', 'USB', 'ALL', 'VRTX', 'MRSH', 'COFA', 'NOC', 'ESS', 'PWR', 'HSY', 'SNPS', 'LVS', 'PPG', 'HWM', 'TRGP', 'EDEN', 'PGR', 'GOOGL', 'HII', 'KMI', 'URW', 'CLARI', 'CAT', 'RNO', 'WDAY', 'ABNB', 'FITB', 'PUB', 'CRUDP', 'CRM', 'ZTS', 'CPAY', 'EW', 'ADBE', 'CPT', 'IVZ', 'CMCSA', 'DECK', 'AIZ', 'NEM', 'VRSK', 'FTNT', 'ALLE', 'COIN', 'SO', 'MMM', 'VST', 'ALCBI', 'TYL', 'OXY', 'ALMKT', 'ALROC', 'PG', 'CRL', 'TDY', 'NXPI', 'CMI', 'FOX', 'ALCLS', 'NANO', 'CRWD', 'VLA', 'NSC', 'ENX', 'DG', 'EXC', 'PHM', 'TEL', 'CMG', 'MPC', 'PFG', 'EXE', 'MCO', 'WDC', 'SLB', 'DDOG']

, 'ALGID'
'BOL',
'AF',
'RUI',
'ALHAF',
'MEDCL',
'AKE',
'BF.B',
'STLAP',
'AYV',
'ELIOR',
'ALTHO',
'GET', 
'GNFT', 
'ALTME',
'STMPA',
 'FRVIA',
 'MERY',
 'ALDBT',
 'SCR',
 'LR',
 'FGR', 
 'NXI', 
 'BRK.B',
 'NACON', 
 'CSCO',
 'ETL', 
 'ERF', 
 'AL2SI',  
 'ALO', 
 'VK', 
 'SPIE', 
 'FDJU',
 'CS', 
 'ALKAL', 
 'MLGEQ', 
 'ALEUP',  
 'TEP',
 'LAT', 
 'EAPI', 
 'SIGHT', 
 'ALALO', 
 'UBI',
 'ALHG', 
 'ALCOX', 
 'ALCPB', 
 'ALSGD', 
 'ALPWG',
 'ALAGO', 
 'BVI', 
 'COFA', 
 'URW', 
 'CLARI', 
 'RNO', 
 'PUB', 
 'CRUDP', 
 'ALCBI', 
 'ALMKT',
 'ALROC', 
 'ALCLS',
 'NANO', 
 'VLA', 
 'ENX', 
 'ALTO'
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


