from pathlib import Path

WORKING_DIR = Path.cwd()

CONFIG_DICT = {"datasets": 
                        {
                        "electricity" : WORKING_DIR / "data/electricity/",
                        "retail"      : WORKING_DIR / "data/retail/",
                        },
               "models":
                        {
                         "electricity"         : WORKING_DIR / "models/electricity/",
                         "retail"              : WORKING_DIR / "models/retail/",
                        },
               }
