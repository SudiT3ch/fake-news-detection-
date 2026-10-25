"""Usage: python predict.py "news headline or article text" """
import sys
from src.prediction import predict_news

if len(sys.argv) < 2:
    sys.exit('Usage: python predict.py "news text"')
try:
    r = predict_news(" ".join(sys.argv[1:]))
except (ValueError, FileNotFoundError) as e:
    sys.exit(f"Error: {e}")
print(f"Prediction: {r['label']}" + (f" (model probability {r['probability']:.2f})" if r["probability"] is not None else ""))
print(r["disclaimer"])
