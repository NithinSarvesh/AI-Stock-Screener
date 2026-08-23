import yfinance as yf

from v6_inference import PPOV6Inference


TICKER = "RELIANCE.NS"


print("=" * 70)
print("PPO V6 INFERENCE TEST")
print("=" * 70)

print("\nDownloading:", TICKER)

df = yf.Ticker(TICKER).history(
    period="1y",
    interval="1d",
    auto_adjust=True,
    actions=False,
)

if df.empty:
    raise RuntimeError("No market data downloaded.")

print("Raw rows:", len(df))

model = PPOV6Inference()

result = model.predict_from_dataframe(
    df,
    current_position=0.0,
)

print("\n" + "=" * 70)
print("PPO V6 RESULT")
print("=" * 70)

print("Action ID      :", result["action_id"])
print("Action         :", result["name"])
print("Position       :", result["position"])
print("Current price  :", result["current_price"])
print("Observation    :", result["observation_size"])
print("Model          :", result["model"])

print("\nObservation:")
print(result["observation"])

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)