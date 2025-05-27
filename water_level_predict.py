import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# historical data
data = {
    "ds": pd.date_range(start="2024-01-01", periods=52, freq='W'),
    "y": [
        114.70, 114.69, 114.68, 114.67, 114.66, 114.65, 114.64, 114.63, 114.62, 114.61,
        114.60, 114.61, 114.62, 114.64, 114.66, 114.68, 114.70, 114.72, 114.74, 114.76,
        114.78, 114.80, 114.79, 114.78, 114.77, 114.76, 114.75, 114.74, 114.73, 114.72,
        114.71, 114.70, 114.69, 114.68, 114.67, 114.66, 114.65, 114.64, 114.63, 114.62,
        114.61, 114.60, 114.59, 114.58, 114.57, 114.56, 114.54, 114.53, 114.53, 114.54,
        114.55, 114.56
    ]
}

df = pd.DataFrame(data)

# Initialize and train the Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='additive'
)
model.fit(df)

# DataFrame for future predictions (next 52 weeks)
future = model.make_future_dataframe(periods=52, freq='W')
forecast = model.predict(future)

# results
model.plot(forecast)
plt.title("Prognozowany poziom wody dla jeziora Tałty")
plt.xlabel("Data")
plt.ylabel("Poziom wody (m n.p.m.)")
plt.tight_layout()
plt.show()