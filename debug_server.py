import urllib.request
import json
import sys

API_URL = "http://localhost:8000"

def test_history(history, name):
    print(f"Testing sequence: {name} -> {history}")
    try:
        req = urllib.request.Request(
            f"{API_URL}/analyze/calculate_state",
            data=json.dumps({"history": history}).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as res:
            if res.status != 200:
                 print(f"FAILED (Status {res.status})")
            else:
                 data = json.loads(res.read().decode('utf-8'))
                 print(f"SUCCESS. State keys: {data.keys()}")
                 print(f"Legal actions: {data.get('legal_actions')}")
                 print(f"Is Finished: {data.get('is_finished')}")
    except urllib.error.HTTPError as e:
        print(f"HTTP ERROR {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"ERROR: {e}")
    print("-" * 20)

# Test cases
test_history([], "Empty")
test_history([1], "P0 Check/Call") # CALL=1
test_history([2], "P0 Bet/Raise") # RAISE=2
test_history([1, 1], "P0 Call, P1 Call")
test_history([2, 1], "P0 Raise, P1 Call")
test_history([2, 2], "P0 Raise, P1 Raise") # Re-raise
test_history([2, 2, 1], "P0 Raise, P1 Raise, P0 Call")
test_history([0], "P0 Fold") # FOLD=0
