import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

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

# Initialize Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    interval_width=0.6
)
model.fit(df)

# DataFrame for future predictions (next 52 weeks)
future = model.make_future_dataframe(periods=52, freq='W')
forecast = model.predict(future)

# Create ENHANCED plot
fig, ax = plt.subplots(figsize=(14, 8))

# Split data into historical and predicted
historical_end = len(df)
historical_forecast = forecast[:historical_end]
predicted_forecast = forecast[historical_end-1:]

color_historical = '#2E86AB'
color_predicted = '#A23B72'
color_confidence = '#F18F01'
color_actual = '#C73E1D'

# confidence intervals for predictions only
ax.fill_between(predicted_forecast['ds'], 
                predicted_forecast['yhat_lower'], 
                predicted_forecast['yhat_upper'], 
                alpha=0.2, color=color_confidence, 
                label='Przedział ufności (prognoza)')

# historical actual data points
ax.scatter(df['ds'], df['y'], 
          color=color_actual, s=20, alpha=0.8, 
          label='Dane rzeczywiste', zorder=5)

# historical fitted line
ax.plot(historical_forecast['ds'], historical_forecast['yhat'], 
        color=color_historical, linewidth=2.5, 
        label='Dopasowanie modelu', alpha=0.9)

# predicted line
ax.plot(predicted_forecast['ds'], predicted_forecast['yhat'], 
        color=color_predicted, linewidth=2.5, linestyle='--',
        label='Prognoza (52 tygodnie)', alpha=0.9)

# Add vertical line to separate historical from predicted data
separation_date = df['ds'].iloc[-1]
ax.axvline(x=separation_date, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
ax.text(separation_date, ax.get_ylim()[1]*0.993, 'Koniec danych rzeczywistych', 
        rotation=90, ha='right', va='top', fontsize=9, alpha=0.7)

# Horizontal reference lines for context
mean_level = df['y'].mean()
max_level = df['y'].max()
min_level = df['y'].min()

# Green avg level label
ax.axhline(y=mean_level, color='green', linestyle=':', alpha=0.5, linewidth=1)
ax.text(ax.get_xlim()[0], mean_level + 0.25, f'Średni poziom: {mean_level:.2f}m', 
        fontsize=9, alpha=0.7, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))

ax.set_title('Prognoza poziomu wody jeziora Tałty\nMazury, Polska', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Data', fontsize=12, fontweight='bold')
ax.set_ylabel('Poziom wody (m n.p.m.)', fontsize=12, fontweight='bold')

# Customize y-axis
y_min, y_max = ax.get_ylim()
tick_start = np.floor(y_min * 5) / 5
tick_end = np.ceil(y_max * 5) / 5

ax.set_yticks(np.arange(tick_start, tick_end + 0.01, 0.2))

ax.tick_params(axis='x', rotation=45)
fig.autofmt_xdate()

# Add legend label
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
          fontsize=10, bbox_to_anchor=(0.02, 0.98))

# Add custom grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Add statistics label
stats_text = f"""Statystyki danych historycznych:
Średnia: {mean_level:.2f}m
Maksimum: {max_level:.2f}m
Minimum: {min_level:.2f}m
Zakres: {max_level - min_level:.2f}m"""

ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
        fontsize=9)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# background color
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

plt.show()

print("\n" + "="*60)
print("ANALIZA PROGNOZY POZIOMU WODY JEZIORA TAŁTY")
print("="*60)

last_known = df['y'].iloc[-1]
predicted_end = forecast['yhat'].iloc[-1]
change = predicted_end - last_known

print(f"Ostatni znany poziom wody: {last_known:.2f}m")
print(f"Prognozowany poziom za 52 tygodnie: {predicted_end:.2f}m")
print(f"Przewidywana zmiana: {change:+.2f}m")

if abs(change) > 0.1:
    trend = "wzrostowy" if change > 0 else "spadkowy"
    print(f"Trend: {trend} ({abs(change):.2f}m)")
else:
    print("Trend: stabilny")

print(f"Przedział ufności: {forecast['yhat_lower'].iloc[-1]:.2f}m - {forecast['yhat_upper'].iloc[-1]:.2f}m")
print("="*60)