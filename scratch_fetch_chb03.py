import sys
sys.path.insert(0, ".")
from datasets.epilepsy import CHBMIT

ds = CHBMIT()
print("Fetching subject 3 raw data (download-only, no windowing/training)...", flush=True)
data = ds.get_data(subjects=[3])
n_sessions = sum(len(v) for v in data[3].values()) if 3 in data else 0
print(f"done. subject 3: {len(data.get(3, {}))} session(s), {n_sessions} run(s) total.", flush=True)
