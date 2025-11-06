"""
Simple end-to-end checker for frontend and backend.
Tries frontend on ports 3000 and 3001, and calls backend on port 8000.
Prints concise results.
"""
import sys
import json
import time
from urllib import request, error

FRONTEND_PORTS = [3000, 3001]
BACKEND_URL = 'http://127.0.0.1:8000'


def http_get(url, timeout=5):
    try:
        with request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return None, str(e).encode('utf-8')


def http_post_json(url, payload, timeout=8):
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return None, str(e).encode('utf-8')


if __name__ == '__main__':
    print('Checking backend at', BACKEND_URL)
    for path in ['/health', '/models', '/sample']:
        url = BACKEND_URL + path
        status, body = http_get(url)
        print(f'GET {path} ->', status)

    # test predict
    sample_payload = {
        "features": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189],
        "model_type": "stacking",
        "explain": False
    }
    status, body = http_post_json(BACKEND_URL + '/predict', sample_payload)
    print('POST /predict ->', status)
    try:
        print(json.dumps(json.loads(body), indent=2))
    except Exception:
        print(body[:400])

    # Try frontend root
    ok_port = None
    for p in FRONTEND_PORTS:
        url = f'http://127.0.0.1:{p}/'
        status, body = http_get(url)
        print(f'GET frontend {p} ->', status)
        if status == 200:
            ok_port = p
            break

    if ok_port:
        print('Frontend appears up at port', ok_port)
        # try demo page
        status, body = http_get(f'http://127.0.0.1:{ok_port}/demo')
        print('GET /demo ->', status)
    else:
        print('Frontend not responding on ports 3000/3001')

    print('E2E check complete')
