# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:58:20 2026

@author: grabe
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:30:18 2026

@author: grabe
"""

import os
import warnings
import pandas as pd
import yfinance as yf
import ta

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

START_DATE   = "2021-01-01"


VIX_TICKER   = "^VIX"


# ─── INDEX DEFINITIONS ───────────────────────────────────────────────────────


INDICES = {
    "cac40": {
        "label":      "CAC40",
        "ticker":     "^FCHI",
        "output_dir": "cac40_data",
        "stocks": {
            "Air Liquide":               "AI.PA",
            "Airbus":                    "AIR.PA",
            "ArcelorMittal":             "MT.AS",
            "AXA":                       "CS.PA",
            "BNP Paribas":               "BNP.PA",
            "Bouygues":                  "EN.PA",
            "Capgemini":                 "CAP.PA",
            "Carrefour":                 "CA.PA",
            "Saint-Gobain":              "SGO.PA",
            "Credit Agricole":           "ACA.PA",
            "Danone":                    "BN.PA",
            "Dassault Systemes":         "DSY.PA",
            "Engie":                     "ENGI.PA",
            "EssilorLuxottica":          "EL.PA",
            "Eurofins Scientific":       "ERF.PA",
            "Hermes":                    "RMS.PA",
            "Kering":                    "KER.PA",
            "Legrand":                   "LR.PA",
            "LOreal":                    "OR.PA",
            "LVMH":                      "MC.PA",
            "Michelin":                  "ML.PA",
            "Orange":                    "ORA.PA",
            "Pernod Ricard":             "RI.PA",
            "Publicis":                  "PUB.PA",
            "Renault":                   "RNO.PA",
            "Safran":                    "SAF.PA",
            "Sanofi":                    "SAN.PA",
            "Schneider Electric":        "SU.PA",
            "Societe Generale":          "GLE.PA",
            "Stellantis":                "STLAM.MI",
            "STMicroelectronics":        "STMPA.PA",
            "Teleperformance":           "TEP.PA",
            "Thales":                    "HO.PA",
            "TotalEnergies":             "TTE.PA",
            "Unibail-Rodamco-Westfield": "URW.PA",
            "Veolia":                    "VIE.PA",
            "Vinci":                     "DG.PA",
            "Vivendi":                   "VIV.PA",
            "Worldline":                 "WLN.PA",
            "Alstom":                    "ALO.PA",
        },
    },

    "sp500": {
        "label":      "SP500",
        "ticker":     "^GSPC",
        "output_dir": "sp500_data",
        "stocks": {
            "3M":                        "MMM",
            "Abbott Laboratories":       "ABT",
            "AbbVie":                    "ABBV",
            "Accenture":                 "ACN",
            "Adobe":                     "ADBE",
            "Advanced Micro Devices":    "AMD",
            "AES":                       "AES",
            "Agilent Technologies":      "A",
            "Air Products":              "APD",
            "Airbnb":                    "ABNB",
            "Akamai":                    "AKAM",
            "Albemarle":                 "ALB",
            "Align Technology":          "ALGN",
            "Allegion":                  "ALLE",
            "Allstate":                  "ALL",
            "Alphabet":                  "GOOGL",
            "Amazon":                    "AMZN",
            "Amcor":                     "AMCR",
            "Ameren":                    "AEE",
            "American Airlines":         "AAL",
            "American Electric Power":   "AEP",
            "American Express":          "AXP",
            "American International Group": "AIG",
            "American Tower":            "AMT",
            "American Water Works":      "AWK",
            "Ameriprise Financial":      "AMP",
            "AMETEK":                    "AME",
            "Amgen":                     "AMGN",
            "Amphenol":                  "APH",
            "Analog Devices":            "ADI",
            "Aon":                       "AON",
            "APA":                       "APA",
            "Apple":                     "AAPL",
            "Applied Materials":         "AMAT",
            "Aptiv":                     "APTV",
            "Arch Capital":              "ACGL",
            "Archer-Daniels-Midland":    "ADM",
            "Arista Networks":           "ANET",
            "Arthur J Gallagher":        "AJG",
            "Assurant":                  "AIZ",
            "AT&T":                      "T",
            "Atmos Energy":              "ATO",
            "Autodesk":                  "ADSK",
            "Automatic Data Processing": "ADP",
            "AutoNation":                "AN",
            "AutoZone":                  "AZO",
            "AvalonBay Communities":     "AVB",
            "Avery Dennison":            "AVY",
            "Axon Enterprise":           "AXON",
            "Baker Hughes":              "BKR",
            "Ball":                      "BALL",
            "Bank of America":           "BAC",
            "Bank of New York Mellon":   "BK",
            "Bath & Body Works":         "BBWI",
            "Baxter International":      "BAX",
            "Becton Dickinson":          "BDX",
            "Berkshire Hathaway":        "BRK-B",
            "Bio-Rad":                   "BIO",
            "Bio-Techne":                "TECH",
            "Biogen":                    "BIIB",
            "BlackRock":                 "BLK",
            "Boeing":                    "BA",
            "Booking Holdings":          "BKNG",
            "BorgWarner":                "BWA",
            "Boston Properties":         "BXP",
            "Boston Scientific":         "BSX",
            "Bristol-Myers Squibb":      "BMY",
            "Broadcom":                  "AVGO",
            "Broadridge Financial":      "BR",
            "Brown & Brown":             "BRO",
            "Brown-Forman":              "BF-B",
            "Builders FirstSource":      "BLDR",
            "Bunge Global":              "BG",
            "Bureau Veritas":            "BV",
            "Cadence Design Systems":    "CDNS",
            "Camden Property":           "CPT",
            "Campbell Soup":             "CPB",
            "Capital One":               "COF",
            "Cardinal Health":           "CAH",
            "CarMax":                    "KMX",
            "Carnival":                  "CCL",
            "Carrier Global":            "CARR",
            "Caterpillar":               "CAT",
            "Cboe Global Markets":       "CBOE",
            "CBRE Group":                "CBRE",
            "CDW":                       "CDW",
            "Celanese":                  "CE",
            "Centene":                   "CNC",
            "CenterPoint Energy":        "CNP",
            "CF Industries":             "CF",
            "Charles River Labs":        "CRL",
            "Charles Schwab":            "SCHW",
            "Charter Communications":    "CHTR",
            "Chevron":                   "CVX",
            "Chipotle":                  "CMG",
            "Church & Dwight":           "CHD",
            "Cigna":                     "CI",
            "Cincinnati Financial":      "CINF",
            "Cintas":                    "CTAS",
            "Cisco":                     "CSCO",
            "Citigroup":                 "C",
            "Citizens Financial":        "CFG",
            "Clorox":                    "CLX",
            "CME Group":                 "CME",
            "CMS Energy":                "CMS",
            "Coca-Cola":                 "KO",
            "Cognizant":                 "CTSH",
            "Colgate-Palmolive":         "CL",
            "Comcast":                   "CMCSA",
            "Copart":                    "CPRT",
            "Corning":                   "GLW",
            "Corteva":                   "CTVA",
            "CoStar Group":              "CSGP",
            "Costco":                    "COST",
            "Coterra Energy":            "CTRA",
            "Crown Castle":              "CCI",
            "CSX":                       "CSX",
            "Cummins":                   "CMI",
            "CVS Health":                "CVS",
            "D.R. Horton":               "DHI",
            "Danaher":                   "DHR",
            "Darden Restaurants":        "DRI",
            "DaVita":                    "DVA",
            "Day One Biopharmaceuticals":"DAWN",
            "Deckers Outdoor":           "DECK",
            "Deere":                     "DE",
            "Delta Air Lines":           "DAL",
            "Devon Energy":              "DVN",
            "Dexcom":                    "DXCM",
            "Digital Realty":            "DLR",
            "Dollar General":            "DG",
            "Dollar Tree":               "DLTR",
            "Dominion Energy":           "D",
            "Dover":                     "DOV",
            "Dow":                       "DOW",
            "DTE Energy":                "DTE",
            "Duke Energy":               "DUK",
            "Eastman Chemical":          "EMN",
            "Eaton":                     "ETN",
            "eBay":                      "EBAY",
            "Ecolab":                    "ECL",
            "Edison International":      "EIX",
            "Edwards Lifesciences":      "EW",
            "Electronic Arts":           "EA",
            "Elevance Health":           "ELV",
            "Emerson Electric":          "EMR",
            "Entergy":                   "ETR",
            "EOG Resources":             "EOG",
            "EPAM Systems":              "EPAM",
            "Equifax":                   "EFX",
            "Equinix":                   "EQIX",
            "Equity Residential":        "EQR",
            "Essex Property":            "ESS",
            "Estee Lauder":              "EL",
            "Etsy":                      "ETSY",
            "Eversource Energy":         "ES",
            "Exelon":                    "EXC",
            "Expedia":                   "EXPE",
            "Expeditors International":  "EXPD",
            "Extra Space Storage":       "EXR",
            "ExxonMobil":                "XOM",
            "F5":                        "FFIV",
            "FactSet Research":          "FDS",
            "Fastenal":                  "FAST",
            "Federal Realty":            "FRT",
            "FedEx":                     "FDX",
            "Fidelity National Info":    "FIS",
            "Fifth Third Bancorp":       "FITB",
            "First Solar":               "FSLR",
            "FMC":                       "FMC",
            "Ford Motor":                "F",
            "Fortinet":                  "FTNT",
            "Fortive":                   "FTV",
            "Fox":                       "FOXA",
            "Franklin Resources":        "BEN",
            "Freeport-McMoRan":          "FCX",
            "Garmin":                    "GRMN",
            "Gartner":                   "IT",
            "GE Aerospace":              "GE",
            "GE HealthCare":             "GEHC",
            "Gen Digital":               "GEN",
            "Generac":                   "GNRC",
            "General Dynamics":          "GD",
            "General Mills":             "GIS",
            "General Motors":            "GM",
            "Genuine Parts":             "GPC",
            "Gilead Sciences":           "GILD",
            "Global Payments":           "GPN",
            "Globe Life":                "GL",
            "Goldman Sachs":             "GS",
            "Halliburton":               "HAL",
            "Hartford Financial":        "HIG",
            "Hasbro":                    "HAS",
            "HCA Healthcare":            "HCA",
            "Healthpeak Properties":     "DOC",
            "Henry Schein":              "HSIC",
            "Hershey":                   "HSY",
            "Hewlett Packard Enterprise":"HPE",
            "Hilton Worldwide":          "HLT",
            "Hologic":                   "HOLX",
            "Home Depot":                "HD",
            "Honeywell":                 "HON",
            "Hormel Foods":              "HRL",
            "Host Hotels":               "HST",
            "Howmet Aerospace":          "HWM",
            "HP":                        "HPQ",
            "Hubbell":                   "HUBB",
            "Humana":                    "HUM",
            "Huntington Bancshares":     "HBAN",
            "Huntington Ingalls":        "HII",
            "IDEX":                      "IEX",
            "Illinois Tool Works":       "ITW",
            "Illumina":                  "ILMN",
            "Incyte":                    "INCY",
            "Ingersoll Rand":            "IR",
            "Installed Building Products":"IBP",
            "Insulet":                   "PODD",
            "Intel":                     "INTC",
            "Intercontinental Exchange": "ICE",
            "International Flavors":     "IFF",
            "International Paper":       "IP",
            "Intuit":                    "INTU",
            "Intuitive Surgical":        "ISRG",
            "Invesco":                   "IVZ",
            "Iron Mountain":             "IRM",
            "J.B. Hunt Transport":       "JBHT",
            "Jack Henry & Associates":   "JKHY",
            "Jabil":                     "JBL",
            "Johnson & Johnson":         "JNJ",
            "Johnson Controls":          "JCI",
            "JPMorgan Chase":            "JPM",
            "Keurig Dr Pepper":          "KDP",
            "KeyCorp":                   "KEY",
            "Keysight Technologies":     "KEYS",
            "Kimberly-Clark":            "KMB",
            "Kimco Realty":              "KIM",
            "Kinder Morgan":             "KMI",
            "KLA":                       "KLAC",
            "L3Harris Technologies":     "LHX",
            "Labcorp":                   "LH",
            "Lam Research":              "LRCX",
            "Lamb Weston":               "LW",
            "Las Vegas Sands":           "LVS",
            "Leidos":                    "LDOS",
            "Lennar":                    "LEN",
            "Lincoln National":          "LNC",
            "LKQ":                       "LKQ",
            "Lockheed Martin":           "LMT",
            "Loews":                     "L",
            "Lowe's":                    "LOW",
            "LPL Financial":             "LPLA",
            "Lumen Technologies":        "LUMN",
            "LyondellBasell":            "LYB",
            "Mara Holdings":             "MARA",
            "Marathon Petroleum":        "MPC",
            "MarketAxess":               "MKTX",
            "Marriott International":    "MAR",
            "Masco":                     "MAS",
            "Mastercard":                "MA",
            "McCormick":                 "MKC",
            "McDonald's":                "MCD",
            "McKesson":                  "MCK",
            "Medtronic":                 "MDT",
            "Merck":                     "MRK",
            "Meta Platforms":            "META",
            "MetLife":                   "MET",
            "Mettler-Toledo":            "MTD",
            "Microsoft":                 "MSFT",
            "Mid-America Apartment":     "MAA",
            "Moderna":                   "MRNA",
            "Mohawk Industries":         "MHK",
            "Molina Healthcare":         "MOH",
            "Molson Coors":              "TAP",
            "Mondelez":                  "MDLZ",
            "MongoDB":                   "MDB",
            "Monolithic Power Systems":  "MPWR",
            "Monster Beverage":          "MNST",
            "Moody's":                   "MCO",
            "Morgan Stanley":            "MS",
            "Mosaic":                    "MOS",
            "MSCI":                      "MSCI",
            "Nasdaq":                    "NDAQ",
            "NetApp":                    "NTAP",
            "Netflix":                   "NFLX",
            "Newmont":                   "NEM",
            "News Corp":                 "NWSA",
            "NextEra Energy":            "NEE",
            "Nike":                      "NKE",
            "NiSource":                  "NI",
            "Norfolk Southern":          "NSC",
            "Northern Trust":            "NTRS",
            "Northrop Grumman":          "NOC",
            "Norwegian Cruise Line":     "NCLH",
            "NRG Energy":                "NRG",
            "Nucor":                     "NUE",
            "Nvidia":                    "NVDA",
            "NVR":                       "NVR",
            "Occidental Petroleum":      "OXY",
            "Omnicom":                   "OMC",
            "ON Semiconductor":          "ON",
            "Oracle":                    "ORCL",
            "Otis Worldwide":            "OTIS",
            "Packaging Corp":            "PKG",
            "Palo Alto Networks":        "PANW",
            "Parker Hannifin":           "PH",
            "Paychex":                   "PAYX",
            "PayPal":                    "PYPL",
            "Penn Entertainment":        "PENN",
            "Pentair":                   "PNR",
            "PepsiCo":                   "PEP",
            "Pfizer":                    "PFE",
            "Philip Morris":             "PM",
            "Phillips 66":               "PSX",
            "PNC Financial":             "PNC",
            "Pool":                      "POOL",
            "PPG Industries":            "PPG",
            "PPL":                       "PPL",
            "Principal Financial":       "PFG",
            "Procter & Gamble":          "PG",
            "Progressive":               "PGR",
            "Prologis":                  "PLD",
            "Prudential Financial":      "PRU",
            "Public Storage":            "PSA",
            "PulteGroup":                "PHM",
            "Qualcomm":                  "QCOM",
            "Quest Diagnostics":         "DGX",
            "Raymond James":             "RJF",
            "Realty Income":             "O",
            "Regency Centers":           "REG",
            "Regeneron":                 "REGN",
            "Republic Services":         "RSG",
            "ResMed":                    "RMD",
            "Revvity":                   "RVTY",
            "Robert Half":               "RHI",
            "Rockwell Automation":       "ROK",
            "Rollins":                   "ROL",
            "Royal Caribbean":           "RCL",
            "RTX":                       "RTX",
            "S&P Global":                "SPGI",
            "Salesforce":                "CRM",
            "SBA Communications":        "SBAC",
            "Schlumberger":              "SLB",
            "Seagate Technology":        "STX",
            "Sempra":                    "SRE",
            "ServiceNow":                "NOW",
            "Sherwin-Williams":          "SHW",
            "Simon Property Group":      "SPG",
            "Skyworks Solutions":        "SWKS",
            "Snap-on":                   "SNA",
            "Solventum":                 "SOLV",
            "Southern Company":          "SO",
            "Southwest Airlines":        "LUV",
            "Stanley Black & Decker":    "SWK",
            "Starbucks":                 "SBUX",
            "State Street":              "STT",
            "Steel Dynamics":            "STLD",
            "Steris":                    "STE",
            "Stryker":                   "SYK",
            "Super Micro Computer":      "SMCI",
            "Suzuki Motor":              "SZKMY",
            "Synchrony Financial":       "SYF",
            "Synopsys":                  "SNPS",
            "sysco":                     "SYY",
            "T-Mobile":                  "TMUS",
            "T. Rowe Price":             "TROW",
            "Take-Two Interactive":      "TTWO",
            "Tapestry":                  "TPR",
            "Targa Resources":           "TRGP",
            "Target":                    "TGT",
            "TE Connectivity":           "TEL",
            "Teradyne":                  "TER",
            "Tesla":                     "TSLA",
            "Texas Instruments":         "TXN",
            "Textron":                   "TXT",
            "Thermo Fisher Scientific":  "TMO",
            "Tractor Supply":            "TSCO",
            "Travelers":                 "TRV",
            "Trane Technologies":        "TT",
            "TransDigm Group":           "TDG",
            "Truist Financial":          "TFC",
            "Tyler Technologies":        "TYL",
            "Uber":                      "UBER",
            "UDR":                       "UDR",
            "Ulta Beauty":               "ULTA",
            "Union Pacific":             "UNP",
            "United Airlines":           "UAL",
            "United Parcel Service":     "UPS",
            "United Rentals":            "URI",
            "UnitedHealth Group":        "UNH",
            "US Bancorp":                "USB",
            "Valero Energy":             "VLO",
            "Ventas":                    "VTR",
            "Verisign":                  "VRSN",
            "Verisk Analytics":          "VRSK",
            "Verizon":                   "VZ",
            "Vertex Pharmaceuticals":    "VRTX",
            "Visa":                      "V",
            "Vistra":                    "VST",
            "Vulcan Materials":          "VMC",
            "W.R. Berkley":              "WRB",
            "W.W. Grainger":             "GWW",
            "Walmart":                   "WMT",
            "Walt Disney":               "DIS",
            "Waters":                    "WAT",
            "Welltower":                 "WELL",
            "Wells Fargo":               "WFC",
            "West Pharmaceutical":       "WST",
            "Western Digital":           "WDC",
            "Western Union":             "WU",
            "Westinghouse Air Brake":    "WAB",
            "Weyerhaeuser":              "WY",
            "Williams Companies":        "WMB",
            "Willis Towers Watson":      "WTW",
            "Workday":                   "WDAY",
            "Xcel Energy":               "XEL",
            "Xylem":                     "XYL",
            "Yum! Brands":               "YUM",
            "Zebra Technologies":        "ZBRA",
            "Zimmer Biomet":             "ZBH",
            "Zoetis":                    "ZTS",
        },
    },

    "sti": {
        "label":      "STI",
        "ticker":     "^STI",
        "output_dir": "sti_data",
        "stocks": {
            "DBS Group":                 "D05.SI",
            "OCBC Bank":                 "O39.SI",
            "UOB":                       "U11.SI",
            "Singapore Telecom":         "Z74.SI",
            "Jardine Matheson":          "J36.SI",
            "Jardine Cycle & Carriage":  "C07.SI",
            "CapitaLand Investment":      "9CI.SI",
            "Wilmar International":      "F34.SI",
            "Keppel":                    "BN4.SI",
            "Singapore Airlines":        "C6L.SI",
            "Genting Singapore":         "G13.SI",
            "City Developments":         "C09.SI",
            "ComfortDelGro":             "C52.SI",
            "Frasers Logistics Trust":   "BUOU.SI",
            "Hongkong Land":             "H78.SI",
            "CapitaLand Ascendas REIT":  "A17U.SI",
            "CapitaLand Integrated Comm":"C38U.SI",
            "Mapletree Pan Asia Comm":   "N2IU.SI",
            "Mapletree Logistics Trust": "M44U.SI",
            "Mapletree Industrial Trust":"ME8U.SI",
            "Thai Beverage":             "Y92.SI",
            "UOL Group":                 "U14.SI",
            "Venture":                   "V03.SI",
            "Yangzijiang Shipbuilding":  "BS6.SI",
            "Sembcorp Industries":       "U96.SI",
            "ST Engineering":            "S63.SI",
            "SATS":                      "S58.SI",
            "Frasers Centrepoint Trust": "J69U.SI",
        },
    },
}

# ─── secteur DEFINITIONS ───────────────────────────────────────────────────────
secteur= {
    "Technology": "IXN",      # iShares Global Tech ETF (proxy MSCI)
    "Healthcare": "IXJ",
    "Financial Services": "IXG",
    "Consumer Defensive": "RXI",
    "Consumer Cyclical": "KXI",
    "Energy": "IXC",
    "Industrials": "EXI",
    "Basic Materials": "MXI",
    "Utilities": "JXI",
    "Real Estate": "REET",
    "Communication Services": "IXP",
    "Unknown":None
}

TICKER_SECTOR={'AI.PA': 'Basic_Materials', 'AIR.PA': 'Industrials', 'MT.AS': 'Basic_Materials', 'CS.PA': 'Financial_Services', 'BNP.PA': 'Financial_Services', 'EN.PA': 'Industrials', 'CAP.PA': 'Technology', 'CA.PA': 'Consumer_Defensive', 'SGO.PA': 'Industrials', 'ACA.PA': 'Financial_Services', 'BN.PA': 'Consumer_Defensive', 'DSY.PA': 'Technology', 'ENGI.PA': 'Utilities', 'EL.PA': 'Healthcare', 'ERF.PA': 'Healthcare', 'RMS.PA': 'Consumer_Cyclical', 'KER.PA': 'Consumer_Cyclical', 'LR.PA': 'Industrials', 'OR.PA': 'Consumer_Defensive', 'MC.PA': 'Consumer_Cyclical', 'ML.PA': 'Unknown', 'ORA.PA': 'Communication_Services', 'RI.PA': 'Consumer_Defensive', 'PUB.PA': 'Communication_Services', 'RNO.PA': 'Consumer_Cyclical', 'SAF.PA': 'Industrials', 'SAN.PA': 'Healthcare', 'SU.PA': 'Industrials', 'GLE.PA': 'Financial_Services', 'STLAM.MI': 'Consumer_Cyclical', 'STMPA.PA': 'Technology', 'TEP.PA': 'Industrials', 'HO.PA': 'Industrials', 'TTE.PA': 'Energy', 'URW.PA': 'Real_Estate', 'VIE.PA': 'Industrials', 'DG.PA': 'Industrials', 'VIV.PA': 'Communication_Services', 'WLN.PA': 'Technology', 'ALO.PA': 'Industrials', 'MMM': 'Industrials', 'ABT': 'Healthcare', 'ABBV': 'Healthcare', 'ACN': 'Technology', 'ADBE': 'Technology', 'AMD': 'Technology', 'AES': 'Utilities', 'A': 'Healthcare', 'APD': 'Basic_Materials', 'ABNB': 'Consumer_Cyclical', 'AKAM': 'Technology', 'ALB': 'Basic_Materials', 'ALGN': 'Healthcare', 'ALLE': 'Industrials', 'ALL': 'Financial_Services', 'GOOGL': 'Communication_Services', 'AMZN': 'Consumer_Cyclical', 'AMCR': 'Consumer_Cyclical', 'AEE': 'Utilities', 'AAL': 'Industrials', 'AEP': 'Utilities', 'AXP': 'Financial_Services', 'AIG': 'Financial_Services', 'AMT': 'Real_Estate', 'AWK': 'Utilities', 'AMP': 'Financial_Services', 'AME': 'Industrials', 'AMGN': 'Healthcare', 'APH': 'Technology', 'ADI': 'Technology', 'ANSS': 'Unknown', 'AON': 'Financial_Services', 'APA': 'Energy', 'AAPL': 'Technology', 'AMAT': 'Technology', 'APTV': 'Consumer_Cyclical', 'ACGL': 'Financial_Services', 'ADM': 'Consumer_Defensive', 'ANET': 'Technology', 'AJG': 'Financial_Services', 'AIZ': 'Financial_Services', 'T': 'Communication_Services', 'ATO': 'Utilities', 'ADSK': 'Technology', 'ADP': 'Technology', 'AN': 'Consumer_Cyclical', 'AZO': 'Consumer_Cyclical', 'AVB': 'Real_Estate', 'AVY': 'Consumer_Cyclical', 'AXON': 'Industrials', 'BKR': 'Energy', 'BALL': 'Consumer_Cyclical', 'BAC': 'Financial_Services', 'BK': 'Financial_Services', 'BBWI': 'Consumer_Cyclical', 'BAX': 'Healthcare', 'BDX': 'Healthcare', 'BRK-B': 'Financial_Services', 'BIO': 'Healthcare', 'TECH': 'Healthcare', 'BIIB': 'Healthcare', 'BLK': 'Financial_Services', 'BA': 'Industrials', 'BKNG': 'Consumer_Cyclical', 'BWA': 'Consumer_Cyclical', 'BXP': 'Real_Estate', 'BSX': 'Healthcare', 'BMY': 'Healthcare', 'AVGO': 'Technology', 'BR': 'Technology', 'BRO': 'Financial_Services', 'BF-B': 'Consumer_Defensive', 'BLDR': 'Industrials', 'BG': 'Consumer_Defensive', 'BV': 'Industrials', 'CDNS': 'Technology', 'CPT': 'Real_Estate', 'CPB': 'Consumer_Defensive', 'COF': 'Financial_Services', 'CAH': 'Healthcare', 'KMX': 'Consumer_Cyclical', 'CCL': 'Consumer_Cyclical', 'CARR': 'Industrials', 'CAT': 'Industrials', 'CBOE': 'Financial_Services', 'CBRE': 'Real_Estate', 'CDW': 'Technology', 'CE': 'Basic_Materials', 'CNC': 'Healthcare', 'CNP': 'Utilities', 'CF': 'Basic_Materials', 'CRL': 'Healthcare', 'SCHW': 'Financial_Services', 'CHTR': 'Communication_Services', 'CVX': 'Energy', 'CMG': 'Consumer_Cyclical', 'CHD': 'Consumer_Defensive', 'CI': 'Healthcare', 'CINF': 'Financial_Services', 'CTAS': 'Industrials', 'CSCO': 'Technology', 'C': 'Financial_Services', 'CFG': 'Financial_Services', 'CLX': 'Consumer_Defensive', 'CME': 'Financial_Services', 'CMS': 'Utilities', 'KO': 'Consumer_Defensive', 'CTSH': 'Technology', 'CL': 'Consumer_Defensive', 'CMCSA': 'Communication_Services', 'CPRT': 'Industrials', 'GLW': 'Technology', 'CTVA': 'Basic_Materials', 'CSGP': 'Real_Estate', 'COST': 'Consumer_Defensive', 'CTRA': 'Energy', 'CCI': 'Real_Estate', 'CSX': 'Industrials', 'CMI': 'Industrials', 'CVS': 'Healthcare', 'DHI': 'Consumer_Cyclical', 'DHR': 'Healthcare', 'DRI': 'Consumer_Cyclical', 'DVA': 'Healthcare', 'DAWN': 'Healthcare', 'DECK': 'Consumer_Cyclical', 'DE': 'Industrials', 'DAL': 'Industrials', 'DVN': 'Energy', 'DXCM': 'Healthcare', 'DLR': 'Real_Estate', 'DFS': 'Unknown', 'DG': 'Consumer_Defensive', 'DLTR': 'Consumer_Defensive', 'D': 'Utilities', 'DOV': 'Industrials', 'DOW': 'Basic_Materials', 'DTE': 'Utilities', 'DUK': 'Utilities', 'EMN': 'Basic_Materials', 'ETN': 'Industrials', 'EBAY': 'Consumer_Cyclical', 'ECL': 'Basic_Materials', 'EIX': 'Utilities', 'EW': 'Healthcare', 'EA': 'Communication_Services', 'ELV': 'Healthcare', 'EMR': 'Industrials', 'ETR': 'Utilities', 'EOG': 'Energy', 'EPAM': 'Technology', 'EFX': 'Industrials', 'EQIX': 'Real_Estate', 'EQR': 'Real_Estate', 'ESS': 'Real_Estate', 'EL': 'Consumer_Defensive', 'ETSY': 'Consumer_Cyclical', 'ES': 'Utilities', 'EXC': 'Utilities', 'EXPE': 'Consumer_Cyclical', 'EXPD': 'Industrials', 'EXR': 'Real_Estate', 'XOM': 'Energy', 'FFIV': 'Technology', 'FDS': 'Financial_Services', 'FAST': 'Industrials', 'FRT': 'Real_Estate', 'FDX': 'Industrials', 'FIS': 'Technology', 'FITB': 'Financial_Services', 'FSLR': 'Technology', 'FI': 'Unknown', 'FLT': 'Unknown', 'FMC': 'Basic_Materials', 'F': 'Consumer_Cyclical', 'FTNT': 'Technology', 'FTV': 'Technology', 'FOXA': 'Communication_Services', 'BEN': 'Financial_Services', 'FCX': 'Basic_Materials', 'GRMN': 'Technology', 'IT': 'Technology', 'GE': 'Industrials', 'GEHC': 'Healthcare', 'GEN': 'Technology', 'GNRC': 'Industrials', 'GD': 'Industrials', 'GIS': 'Consumer_Defensive', 'GM': 'Consumer_Cyclical', 'GPC': 'Consumer_Cyclical', 'GILD': 'Healthcare', 'GPN': 'Industrials', 'GL': 'Financial_Services', 'GS': 'Financial_Services', 'HAL': 'Energy', 'HIG': 'Financial_Services', 'HAS': 'Consumer_Cyclical', 'HCA': 'Healthcare', 'DOC': 'Real_Estate', 'HSIC': 'Healthcare', 'HSY': 'Consumer_Defensive', 'HES': 'Unknown', 'HPE': 'Technology', 'HLT': 'Consumer_Cyclical', 'HOLX': 'Healthcare', 'HD': 'Consumer_Cyclical', 'HON': 'Industrials', 'HRL': 'Consumer_Defensive', 'HST': 'Real_Estate', 'HWM': 'Industrials', 'HPQ': 'Technology', 'HUBB': 'Industrials', 'HUM': 'Healthcare', 'HBAN': 'Financial_Services', 'HII': 'Industrials', 'IEX': 'Industrials', 'ITW': 'Industrials', 'ILMN': 'Healthcare', 'INCY': 'Healthcare', 'IR': 'Industrials', 'IBP': 'Consumer_Cyclical', 'PODD': 'Healthcare', 'INTC': 'Technology', 'ICE': 'Financial_Services', 'IFF': 'Basic_Materials', 'IP': 'Consumer_Cyclical', 'IPG': 'Unknown', 'INTU': 'Technology', 'ISRG': 'Healthcare', 'IVZ': 'Financial_Services', 'IRM': 'Real_Estate', 'JBHT': 'Industrials', 'JKHY': 'Technology', 'JBL': 'Technology', 'JNJ': 'Healthcare', 'JCI': 'Industrials', 'JPM': 'Financial_Services', 'JNPR': 'Unknown', 'K': 'Unknown', 'KDP': 'Consumer_Defensive', 'KEY': 'Financial_Services', 'KEYS': 'Technology', 'KMB': 'Consumer_Defensive', 'KIM': 'Real_Estate', 'KMI': 'Energy', 'KLAC': 'Technology', 'LHX': 'Industrials', 'LH': 'Healthcare', 'LRCX': 'Technology', 'LW': 'Consumer_Defensive', 'LVS': 'Consumer_Cyclical', 'LDOS': 'Technology', 'LEN': 'Consumer_Cyclical', 'LNC': 'Financial_Services', 'LKQ': 'Consumer_Cyclical', 'LMT': 'Industrials', 'L': 'Financial_Services', 'LOW': 'Consumer_Cyclical', 'LPLA': 'Financial_Services', 'LUMN': 'Communication_Services', 'LYB': 'Basic_Materials', 'MARA': 'Financial_Services', 'MRO': 'Unknown', 'MPC': 'Energy', 'MKTX': 'Financial_Services', 'MAR': 'Consumer_Cyclical', 'MMC': 'Unknown', 'MAS': 'Industrials', 'MA': 'Financial_Services', 'MKC': 'Consumer_Defensive', 'MCD': 'Consumer_Cyclical', 'MCK': 'Healthcare', 'MDT': 'Healthcare', 'MRK': 'Healthcare', 'META': 'Communication_Services', 'MET': 'Financial_Services', 'MTD': 'Healthcare', 'MSFT': 'Technology', 'MAA': 'Real_Estate', 'MRNA': 'Healthcare', 'MHK': 'Consumer_Cyclical', 'MOH': 'Healthcare', 'TAP': 'Consumer_Defensive', 'MDLZ': 'Consumer_Defensive', 'MDB': 'Technology', 'MPWR': 'Technology', 'MNST': 'Consumer_Defensive', 'MCO': 'Financial_Services', 'MS': 'Financial_Services', 'MOS': 'Basic_Materials', 'MSCI': 'Financial_Services', 'NDAQ': 'Financial_Services', 'NTAP': 'Technology', 'NFLX': 'Communication_Services', 'NEM': 'Basic_Materials', 'NWSA': 'Communication_Services', 'NEE': 'Utilities', 'NKE': 'Consumer_Cyclical', 'NI': 'Utilities', 'NSC': 'Industrials', 'NTRS': 'Financial_Services', 'NOC': 'Industrials', 'NCLH': 'Consumer_Cyclical', 'NRG': 'Utilities', 'NUE': 'Basic_Materials', 'NVDA': 'Technology', 'NVR': 'Consumer_Cyclical', 'OXY': 'Energy', 'OMC': 'Communication_Services', 'ON': 'Technology', 'ORCL': 'Technology', 'OTIS': 'Industrials', 'PKG': 'Consumer_Cyclical', 'PANW': 'Technology', 'PH': 'Industrials', 'PAYX': 'Technology', 'PYPL': 'Financial_Services', 'PENN': 'Consumer_Cyclical', 'PNR': 'Industrials', 'PEP': 'Consumer_Defensive', 'PFE': 'Healthcare', 'PM': 'Consumer_Defensive', 'PSX': 'Energy', 'PNC': 'Financial_Services', 'POOL': 'Industrials', 'PPG': 'Basic_Materials', 'PPL': 'Utilities', 'PFG': 'Financial_Services', 'PG': 'Consumer_Defensive', 'PGR': 'Financial_Services', 'PLD': 'Real_Estate', 'PRU': 'Financial_Services', 'PSA': 'Real_Estate', 'PHM': 'Consumer_Cyclical', 'QCOM': 'Technology', 'DGX': 'Healthcare', 'RJF': 'Financial_Services', 'O': 'Real_Estate', 'REG': 'Real_Estate', 'REGN': 'Healthcare', 'RSG': 'Industrials', 'RMD': 'Healthcare', 'RVTY': 'Healthcare', 'RHI': 'Industrials', 'ROK': 'Industrials', 'ROL': 'Consumer_Cyclical', 'RCL': 'Consumer_Cyclical', 'RTX': 'Industrials', 'SPGI': 'Financial_Services', 'CRM': 'Technology', 'SBAC': 'Real_Estate', 'SLB': 'Energy', 'STX': 'Technology', 'SRE': 'Utilities', 'NOW': 'Technology', 'SHW': 'Basic_Materials', 'SPG': 'Real_Estate', 'SWKS': 'Technology', 'SNA': 'Industrials', 'SOLV': 'Healthcare', 'SO': 'Utilities', 'LUV': 'Industrials', 'SWK': 'Industrials', 'SBUX': 'Consumer_Cyclical', 'STT': 'Financial_Services', 'STLD': 'Basic_Materials', 'STE': 'Healthcare', 'SYK': 'Healthcare', 'SMCI': 'Technology', 'SZKMY': 'Consumer_Cyclical', 'SYF': 'Financial_Services', 'SNPS': 'Technology', 'SYY': 'Consumer_Defensive', 'TMUS': 'Communication_Services', 'TROW': 'Financial_Services', 'TTWO': 'Communication_Services', 'TPR': 'Consumer_Cyclical', 'TRGP': 'Energy', 'TGT': 'Consumer_Defensive', 'TEL': 'Technology', 'TER': 'Technology', 'TSLA': 'Consumer_Cyclical', 'TXN': 'Technology', 'TXT': 'Industrials', 'TMO': 'Healthcare', 'TSCO': 'Consumer_Cyclical', 'TRV': 'Financial_Services', 'TT': 'Industrials', 'TDG': 'Industrials', 'TFC': 'Financial_Services', 'TYL': 'Technology', 'UBER': 'Technology', 'UDR': 'Real_Estate', 'ULTA': 'Consumer_Cyclical', 'UNP': 'Industrials', 'UAL': 'Industrials', 'UPS': 'Industrials', 'URI': 'Industrials', 'UNH': 'Healthcare', 'USB': 'Financial_Services', 'VLO': 'Energy', 'VTR': 'Real_Estate', 'VRSN': 'Technology', 'VRSK': 'Industrials', 'VZ': 'Communication_Services', 'VRTX': 'Healthcare', 'V': 'Financial_Services', 'VST': 'Utilities', 'VMW': 'Unknown', 'VMC': 'Basic_Materials', 'WRB': 'Financial_Services', 'GWW': 'Industrials', 'WBA': 'Unknown', 'WMT': 'Consumer_Defensive', 'DIS': 'Communication_Services', 'WAT': 'Healthcare', 'WELL': 'Real_Estate', 'WFC': 'Financial_Services', 'WST': 'Healthcare', 'WDC': 'Technology', 'WU': 'Financial_Services', 'WAB': 'Industrials', 'WY': 'Real_Estate', 'WMB': 'Energy', 'WTW': 'Financial_Services', 'GORE': 'Unknown', 'WDAY': 'Technology', 'XEL': 'Utilities', 'XYL': 'Industrials', 'YUM': 'Consumer_Cyclical', 'ZBRA': 'Technology', 'ZBH': 'Healthcare', 'ZTS': 'Healthcare', 'D05.SI': 'Financial_Services', 'O39.SI': 'Financial_Services', 'U11.SI': 'Financial_Services', 'Z74.SI': 'Communication_Services', 'J36.SI': 'Industrials', 'C07.SI': 'Industrials', '9CI.SI': 'Real_Estate', 'F34.SI': 'Consumer_Defensive', 'BN4.SI': 'Industrials', 'C6L.SI': 'Industrials', 'G13.SI': 'Consumer_Cyclical', 'C09.SI': 'Real_Estate', 'C52.SI': 'Industrials', 'BUOU.SI': 'Real_Estate', 'H78.SI': 'Real_Estate', 'A17U.SI': 'Real_Estate', 'C38U.SI': 'Real_Estate', 'N2IU.SI': 'Real_Estate', 'M44U.SI': 'Real_Estate', 'ME8U.SI': 'Real_Estate', 'Y92.SI': 'Consumer_Defensive', 'U14.SI': 'Real_Estate', 'V03.SI': 'Technology', 'BS6.SI': 'Industrials', 'U96.SI': 'Utilities', 'S63.SI': 'Industrials', 'S51.SI': 'Unknown', 'S58.SI': 'Industrials', 'EMF.SI': 'Unknown', 'J69U.SI': 'Real_Estate'}
# ─── HELPERS ─────────────────────────────────────────────────────────────────

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    df["MA20"]          = close.rolling(20).mean()
    df["MA50"]          = close.rolling(50).mean()
    df["Momentum"]      = close.diff(10)
    df["RSI"]           = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]          = macd.macd()
    df["MACD_Signal"]   = macd.macd_signal()
    df["MACD_Hist"]     = macd.macd_diff()

    df["Volatility_20d"] = close.pct_change().rolling(20).std() * (252 ** 0.5)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"]      = bb.bollinger_hband()
    df["BB_Lower"]      = bb.bollinger_lband()
    df["BB_Mid"]        = bb.bollinger_mavg()

    return df


def download_index(label: str, ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
    raw = flatten_columns(raw)
    raw.index = pd.to_datetime(raw.index)
    raw.columns = [f"{label}_{c}" for c in raw.columns]
    return raw

def download_VIX():
    print(f"\nDownloading VIX ({VIX_TICKER}) ...")
    vix_raw   = yf.download(VIX_TICKER, start=START_DATE,
                             auto_adjust=True, progress=False)
    vix_raw   = flatten_columns(vix_raw)
    vix_close = vix_raw["Close"].copy()
    vix_close.name = "VIX_Close"
    vix_close.index = pd.to_datetime(vix_close.index)
    print(f"  {len(vix_close)} trading days fetched.")
    return vix_close

def arboraissance(market):
    
    base_path = os.path.join("data_base", market)
    
    for sec in secteur:
        name = sec.replace(" ", "_")
        output_dir = os.path.join(base_path, name)
        os.makedirs(output_dir, exist_ok=True)
        
def create_csv(df,ticker,output_dir,sector):
    safe_name = ticker.replace(".", "_").replace("^", "").replace("-", "_")
    Path=f"data_base/{output_dir}/{sector}"
    out_path  = os.path.join(Path, f"{safe_name}.csv")
    
    df.index.name = "Date"
    df.sort_index(inplace=True)
    df.to_csv(f"{out_path}")
    print(f"    Saved -> {out_path}  ({len(df)} rows, {len(df.columns)} cols)\n")
    
def all_csv(stocks,idx_df,vix_close,total,output_dir):
    
    tickers = list(stocks.values())

    df_all = yf.download(
        tickers,
        start=START_DATE,
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )
    for i, (company_name, ticker) in enumerate(stocks.items(), 1):
        print(f"  [{i:03d}/{total}] {company_name} ({ticker})")
        if len(tickers) == 1:
            raw = df_all.copy()
        else:
            raw = df_all[ticker].copy()
        # ── Download OHLCV ────────────────────────────────────────────────
        
        sector = TICKER_SECTOR.get(ticker, "Unknown")
        
        # ── Technical indicators ──────────────────────────────────────────
        
        df = compute_indicators(raw.copy())

        # ── Merge own index ───────────────────────────────────────────────
        
        if not idx_df.empty:
            df = df.join(idx_df, how="left")

        # ── Merge VIX ─────────────────────────────────────────────────────
        
        df = df.join(vix_close, how="left")


        # ── Round all numeric columns to 3 decimal places ─────────────────
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].round(3)

        # ── Save ──────────────────────────────────────────────────────────
        create_csv(df,ticker,output_dir,sector)
        
# ─── CORE PROCESSING ─────────────────────────────────────────────────────────
def download_sector_indices(secteur_dict):
    output_dir = os.path.join("data_base", "sector_indices")
    os.makedirs(output_dir, exist_ok=True)

    for sector, ticker in secteur_dict.items():
        print(f"\nDownloading sector: {sector}")

        if not ticker:
            print("  Skipped (no ticker)")
            continue

        try:
            # Téléchargement
            df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
            df = flatten_columns(df)
            df.index = pd.to_datetime(df.index)

            # Calcul indicateurs techniques
            df = compute_indicators(df)

            # Ajout des rendements
            df["Return_1d"] = df["Close"].pct_change()
            df["Return_5d"] = df["Close"].pct_change(5)

            # Normalisation base 100
            df["Normalized"] = df["Close"] / df["Close"].iloc[0] * 100

            # Sauvegarde
            out_path = os.path.join(output_dir, f"{sector.replace(' ', '_')}.csv")
            df.to_csv(out_path)
            print(f"  Saved -> {out_path} ({len(df)} rows, {len(df.columns)} cols)")

        except Exception as e:
            print(f"  Error: {e}")
    for bourse,i in list(INDICES.items()):
        tickers = i["ticker"]
        if not tickers:
            print("  Skipped (no tickers)")
            continue

        try:
            # Téléchargement
            df = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)
            df = flatten_columns(df)
            df.index = pd.to_datetime(df.index)

            # Calcul indicateurs techniques
            df = compute_indicators(df)

            # Ajout des rendements
            df["Return_1d"] = df["Close"].pct_change()
            df["Return_5d"] = df["Close"].pct_change(5)

            # Normalisation base 100
            df["Normalized"] = df["Close"] / df["Close"].iloc[0] * 100

            # Sauvegarde
            out_path = os.path.join(output_dir, f"{bourse}.csv")
            df.to_csv(out_path)
            print(f"  Saved -> {out_path} ({len(df)} rows, {len(df.columns)} cols)")

        except Exception as e:
            print(f"  Error: {e}")
            
def process_index(index_key, cfg, vix_close):

    label= cfg["label"]
    stocks= cfg["stocks"]
    ticker     = cfg["ticker"]     # "^FCHI"
    output_dir = cfg["output_dir"]  # "cac40_data"
    
    arboraissance(output_dir)
        
    print(f"\n{'='*60}")
    print(f"  Processing {label} — {len(stocks)} stocks")
    print(f"  Output dir : ./{output_dir}/")
    print(f"{'='*60}\n")

    # Download own index
    print(f"  Downloading {label} index ({ticker}) ...")
    try:
        idx_df = download_index(label, ticker)
        print(f"  {len(idx_df)} trading days fetched.\n")
    except Exception as e:
        print(f"  [!] Could not download index: {e}. Continuing without it.\n")
        idx_df = pd.DataFrame()

    total = len(stocks)
    all_csv(stocks,idx_df,vix_close,total,output_dir)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # ── CONFIGURE HERE ────────────────────────────────────────────────────────
    
    RUN_INDEX = None
    # ─────────────────────────────────────────────────────────────────────────
    
    download_sector_indices(secteur)
    print("=" * 60)
    print("  Multi-Index Stock Data Fetcher")
    total_stocks = sum(len(v["stocks"]) for v in INDICES.values())
    print(f"  Total stocks : {total_stocks}")
    if RUN_INDEX:
        print(f"  Running only : {RUN_INDEX.upper()}")
    print("=" * 60)

    # Download VIX once — shared across all indices
    vix_close = download_VIX() 

    # Run selected indices
    keys_to_run = [RUN_INDEX] if RUN_INDEX else list(INDICES.keys())
    for key in keys_to_run:
        process_index(key, INDICES[key], vix_close)

    print("\n" + "=" * 60)
    print("  All done!")
    print("  cac40_data/  -> CAC40 stocks")
    print("  sp500_data/  -> S&P 500 stocks")
    print("  sti_data/    -> STI stocks")
    print("=" * 60)


if __name__ == "__main__":
    main()