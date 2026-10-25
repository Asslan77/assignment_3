"""Генератор СИНТЕТИЧЕСКОГО датасета бронирований спортивных площадок (симуляция, не реальные данные).
Выход: data/sports_bookings_synthetic.csv (то, что видит модель) и data/ground_truth.csv (скрытые истинные параметры).
Запуск: python gen_dataset.py
"""
import numpy as np, pandas as pd, json, os
SEED, N_DAYS, N_VENUES = 2026, 730, 14
rng = np.random.default_rng(SEED)
types = ["futsal", "football", "tennis", "swimming"]
base_logit = {"futsal": 0.0, "football": -0.2, "tennis": -0.5, "swimming": -0.3}
beta_type = {"futsal": -2.0, "football": -1.6, "tennis": -1.3, "swimming": -0.9}   # истинный эффект ln r на logit брони
outdoor = {"futsal": 0, "football": 1, "tennis": 1, "swimming": 0}
venues = pd.DataFrame({"venue_id": range(N_VENUES), "venue_type": [types[i % 4] for i in range(N_VENUES)],
                       "ref_price": rng.uniform(3500, 9500, N_VENUES).round(-2), "pop": rng.normal(0, 0.4, N_VENUES)})
days = pd.date_range("2024-09-01", periods=N_DAYS, freq="D"); hours = np.arange(8, 23)
df = pd.MultiIndex.from_product([venues.venue_id, days, hours], names=["venue_id", "day", "hour"]).to_frame(index=False).merge(venues, on="venue_id")
n = len(df)
df["slot_start"] = df["day"] + pd.to_timedelta(df["hour"], unit="h")
doy, dow, month = df.slot_start.dt.dayofyear.values, df.slot_start.dt.dayofweek.values, df.slot_start.dt.month.values
# погода Астаны: примерно от -30 до +35 °C; суточный ход + шум
day_temp = pd.Series(3 + 22 * np.sin(2 * np.pi * (days.dayofyear - 110) / 365) + rng.normal(0, 5, N_DAYS), index=days)
temp_real = df["day"].map(day_temp).values + 4 * np.sin(2 * np.pi * (df.hour.values - 9) / 24) + rng.normal(0, 1.5, n)
precip_real = np.where(rng.uniform(size=n) < 0.18, rng.exponential(0.8, n), 0.0)
wind_real = np.abs(rng.normal(4.5, 2.2, n)); hum_real = np.clip(65 + rng.normal(0, 14, n), 10, 100)
hol_days = pd.Series(rng.uniform(size=N_DAYS) < 0.03, index=days)
df["holiday"] = df["day"].map(hol_days).astype(int)
df["exam"] = (np.isin(month, [1, 5, 12]) & (df.slot_start.dt.day.values > 10)).astype(int)
df["lead_time_h"] = rng.choice([2, 12, 48, 200], n, p=[.25, .35, .3, .1])
peak = ((df.hour >= 18) & (df.hour <= 21)).astype(float).values; weekend = (dow >= 5).astype(float)
out = df.venue_type.map(outdoor).values.astype(float)
logit0 = (-1.2 + df.venue_type.map(base_logit).values + df["pop"].values + 1.5 * peak + 0.6 * weekend + 0.3 * df.holiday.values - 0.25 * df.exam.values
          + out * (0.05 * (np.clip(temp_real, -30, 28) - 10) - 1.0 * np.minimum(precip_real, 3) - 0.05 * np.maximum(wind_real - 8, 0))
          - 0.003 * df.lead_time_h.values + 0.15 * np.sin(2 * np.pi * (df.hour.values - 6) / 24))
beta = df.venue_type.map(beta_type).values * (1 - 0.3 * peak)       # на пике чувствительность к цене ниже
r = np.clip(1 + 0.12 * peak + 0.06 * weekend + 0.03 * df.holiday.values + rng.normal(0, 0.06, n), 0.75, 1.35)  # эндогенная часть + экзогенный шум
df["price"] = (df.ref_price * r).round(0)
r_eff = df.price / df.ref_price
p = 1 / (1 + np.exp(-(logit0 + beta * np.log(r_eff))))
df["booked"] = (rng.uniform(size=n) < p).astype(int)
pc = 0.06 + 0.10 * (r_eff > 1.1) + 0.03 * (df.lead_time_h > 48)
df["cancelled"] = ((rng.uniform(size=n) < pc) & (df.booked == 1)).astype(int)
# в данные идёт ПРОГНОЗ погоды (с ошибкой), а не фактическая погода
df["temp"] = (temp_real + rng.normal(0, 1.5, n)).round(1); df["precip"] = np.maximum(precip_real + rng.normal(0, 0.3, n), 0).round(2)
df["wind"] = np.maximum(wind_real + rng.normal(0, 1.0, n), 0).round(1); df["humidity"] = np.clip(hum_real + rng.normal(0, 5, n), 5, 100).round(0)
df["logit0_true"] = logit0; df["beta_true"] = beta
df = df.sort_values(["slot_start", "venue_id"]).reset_index(drop=True); df["slot_id"] = np.arange(len(df))
os.makedirs("data", exist_ok=True)
cols = ["slot_id", "venue_id", "venue_type", "slot_start", "booked", "price", "ref_price", "cancelled", "temp", "humidity", "wind", "precip", "holiday", "exam", "lead_time_h"]
df[cols].to_csv("data/sports_bookings_synthetic.csv", index=False)
# скрытая истина для проверки пайплайна (в обучение не попадает)
df[["slot_id", "logit0_true", "beta_true"]].to_csv("data/ground_truth.csv", index=False)
json.dump(dict(seed=SEED, n_days=N_DAYS, n_venues=N_VENUES, beta_type=beta_type, note="SYNTHETIC simulation, not real aggregator data"),
          open("data/generation_params.json", "w"), indent=1)
print(len(df), "rows;", "booking rate", round(df.booked.mean(), 3), "| cancel rate among booked", round(df.cancelled.sum() / df.booked.sum(), 3))
