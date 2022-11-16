from pathlib import Path

WORKING_DIR = Path.cwd()

CONFIG_DICT = {"datasets": 
                        {"electricity" : WORKING_DIR / "data/electricity/",
                        "retail"  : WORKING_DIR / "data/retail/",
                        "walmart" : WORKING_DIR / "data/walmart/", 
                        "stocks" : WORKING_DIR / "data/stocks/",
                        },
               "models":
                        {"electricity" : WORKING_DIR / "models/electricity/",
                         "retail"      : WORKING_DIR / "models/retail/",
                         "walmart"     : WORKING_DIR / "models/walmart/",
                         "stocks"      : WORKING_DIR / "models/stocks/",                    
                        }
           }