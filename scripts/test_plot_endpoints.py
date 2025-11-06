"""Quick test of all plot endpoints to identify failures."""
import requests

BASE_URL = "http://localhost:8000"

# All plot endpoints from the screenshot
endpoints = [
    "/plot?type=fi&model=sklearn",
    "/plot?type=roc&model=sklearn",
    "/plot?type=confusion&model=sklearn",
    "/plot?type=pr&model=sklearn",
    "/models",
    "/plot?type=roc&model=stacking",
    "/plot?type=confusion&model=stacking",
    "/plot?type=fi&model=stacking",
    "/plot?type=pr&model=stacking",
    "/plot?type=roc&model=dl",
    "/plot?type=confusion&model=dl",
    "/plot?type=fi&model=dl",
    "/plot?type=pr&model=dl",
]

print("Testing plot endpoints...")
print("=" * 60)

for endpoint in endpoints:
    try:
        resp = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        status = "✓ OK" if resp.status_code == 200 else f"✗ {resp.status_code}"
        print(f"{status:8} GET {endpoint}")
        if resp.status_code != 200:
            try:
                error = resp.json()
                print(f"         Error: {error.get('detail', 'Unknown error')}")
            except:
                print(f"         Raw error: {resp.text[:200]}")
    except Exception as e:
        print(f"✗ ERROR  GET {endpoint}")
        print(f"         Exception: {str(e)[:100]}")

print("=" * 60)
