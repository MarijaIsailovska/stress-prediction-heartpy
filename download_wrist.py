import wfdb
import os

dl_dir = 'data/raw/wrist'
os.makedirs(dl_dir, exist_ok=True)
wfdb.dl_database('wrist', dl_dir)
print("Wrist PPG dataset downloaded!")
