from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import json

from app.main import APP

client = TestClient(APP)

def call_health():
    r = client.get('/health')
    print('health', r.status_code, r.text)


def call_models():
    r = client.get('/models')
    print('/models', r.status_code, r.text)


def call_plot(plot_type='roc', model='sklearn'):
    r = client.get(f'/plot?type={plot_type}&model={model}')
    print(f'/plot?type={plot_type}&model={model}', r.status_code)
    if r.status_code != 200:
        try:
            print('detail:', r.json())
        except Exception:
            print('text:', r.text[:400])
    else:
        print('image bytes:', len(r.content))


def call_predict():
    payload = {
        "features": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189],
        "model_type": "stacking",
        "explain": True
    }
    r = client.post('/predict', json=payload)
    print('/predict', r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print('non-json response:', r.text[:1000])


if __name__ == '__main__':
    call_health()
    call_models()
    call_plot('roc', 'sklearn')
    call_plot('confusion', 'stacking')
    call_predict()
