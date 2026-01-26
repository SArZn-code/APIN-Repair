import logging
from pathlib import Path
from utils import evaluate_classification_metrics
import pandas as pd

def load_cols(csv_path):
    df = pd.read_csv(csv_path)
    y_true = df['Theory'].tolist()
    y_pred = df['Pred'].tolist()
    y_prob = df['Prob'].tolist() if 'Prob' in df.columns else None
    return y_true, y_pred, y_prob

# log recording
log_file = Path('logs/evaluation.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


classifier = 'APIN'
assign = 'iDLM'

before_csv = f'output/prediction_output_{classifier}_{assign}.csv'
after_csv = f'output/repaired_prediction_output_{classifier}_iDLM.csv'

# open csv
before_true, before_pred, before_prob = load_cols(before_csv)
after_true, after_pred, after_prob = load_cols(after_csv)

before_metrics = evaluate_classification_metrics(before_true, before_pred, before_prob)
after_metrics = evaluate_classification_metrics(after_true, after_pred, after_prob)

logger.info('Before metrics:')
for k, v in before_metrics.items():
    logger.info('  %s: %s', k, v)

logger.info('After metrics:')
for k, v in after_metrics.items():
    logger.info('  %s: %s', k, v)