from fastapi.testclient import TestClient
import json
from app.main import APP
import pandas as pd

client = TestClient(APP)


def test_report_endpoint_returns_pdf():
    # load a sample from data.csv to get valid feature values
    df = pd.read_csv('data.csv')
    cols_to_drop = [c for c in df.columns if c.startswith('Unnamed')]
    if 'id' in df.columns:
        cols_to_drop.append('id')
    if 'diagnosis' in df.columns:
        X = df.drop(columns=cols_to_drop + ['diagnosis'])
    else:
        X = df.drop(columns=cols_to_drop)
    sample = X.iloc[0].tolist()

    payload = {"features": sample, "model_type": "sklearn"}
    resp = client.post('/report', json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.headers.get('content-type', '').startswith('application/pdf')
    content = resp.content
    assert content.startswith(b'%PDF'), 'Response is not a PDF'
    assert len(content) > 1000, 'PDF content too small'
