# save as debug.py and run: python debug.py
import json, os

print("Working directory:", os.getcwd())
print("Files here:", os.listdir("."))
print()

for path in ["./IPC_updated.json", "./BNS_updated.json"]:
    exists = os.path.exists(path)
    print(f"{path} exists: {exists}")
    if exists:
        data = json.load(open(path, encoding="utf-8"))
        print(f"  → {len(data)} records, keys: {list(data[0].keys())}")