import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.main import generate_pdf_report_bytes, _render_plot_bytes

DATA = 'data.csv'

print('Reading data...')
df = pd.read_csv(DATA)
cols_to_drop = [c for c in df.columns if c.startswith('Unnamed')]
if 'id' in df.columns:
    cols_to_drop.append('id')
if 'diagnosis' in df.columns:
    X = df.drop(columns=cols_to_drop + ['diagnosis'])
else:
    X = df.drop(columns=cols_to_drop)

sample = X.iloc[0].tolist()
feature_names = list(X.columns)

print('Generating PDF bytes...')
pdf = generate_pdf_report_bytes(sample, feature_names, 'sklearn')
print('PDF bytes length:', len(pdf))

out = 'reports/debug_report_streamed.pdf'
with open(out, 'wb') as f:
    f.write(pdf)
print('Wrote', out)

# Also test individual plot bytes
print('Generating individual ROC plot bytes...')
try:
    pb = _render_plot_bytes('roc', 'sklearn', for_pdf=True)
    print('ROC bytes len:', len(pb))
    with open('reports/debug_roc.png', 'wb') as f:
        f.write(pb)
    print('Wrote reports/debug_roc.png')
except Exception as e:
    print('ROC generation failed:', e)

print('Done')
