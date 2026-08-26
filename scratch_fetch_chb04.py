import sys
sys.path.insert(0, ".")
from datasets.epilepsy import CHBMIT

ds = CHBMIT()
print("Fetching subject 4 raw data (download-only, no windowing/training)...", flush=True)
data = ds.get_data(subjects=[4])
n_sessions = sum(len(v) for v in data[4].values()) if 4 in data else 0
print(f"done. subject 4: {len(data.get(4, {}))} session(s), {n_sessions} run(s) total.", flush=True)
