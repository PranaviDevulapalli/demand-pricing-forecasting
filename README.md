# demand-pricing-forecasting
#  Intelligent Demand Forecasting + Dynamic Pricing

**Machine Learning + Deep Learning Project**  
**By: _Pranavi Devulapalli_**

---

##  Project Summary

This project builds an **end-to-end demand forecasting system** combined with a **simple dynamic pricing engine**.

It uses:

- **Classical ML** → Random Forest Regressor  
- **Deep Learning** → LSTM (Long Short-Term Memory network)

The project demonstrates:

- ✅ How to forecast daily demand using lag-based features  
- ✅ How to compare classical ML vs deep learning for time-series  
- ✅ How to simulate demand as a function of price  
- ✅ How to estimate optimal price using a revenue curve  
- ✅ How to combine forecasting + pricing into a business pipeline  

---

##  1. Problem Statement

Given daily order data, the goal is to predict:

- Future daily demand  
- The best price to charge to maximize revenue  

This mimics how **e-commerce, food delivery, logistics, and retail platforms** make pricing decisions in the real world.

---

##  2. Dataset

Columns included (may vary slightly by version):

- **Daily order counts** (Target)  
- **Urgent / Non-urgent / Returned orders**  
- **Day-of-week, week-of-month**  
- **Lag features**  
- **Rolling averages**

The dataset does **not** include price, so a **simulated economic price–demand model** is used in the pricing step.

---

##  3. Feature Engineering (Core of the Project)

Time-series forecasting requires the model to understand **recent history** and **patterns over time**.

The following features were created:

### ✔ Lag Features

- `lag_1` → yesterday’s demand  
- `lag_7` → demand 1 week ago  

These capture **short and medium-term temporal effects**.

### ✔ Rolling Window Features

- `rolling_mean_7` → average of last 7 days  

This helps:

- Smooth day-to-day noise  
- Capture **trend** and **weekly behavior**

### ✔ Calendar Features

- Day of week  
- Week of month  

These features model **weekly seasonality** (e.g., weekends vs weekdays).

### ✔ Why Feature Engineering Matters

- Tree-based models like **Random Forest** depend heavily on engineered features.  
- **LSTMs** can learn temporal structure internally, but require:
  - Proper scaling  
  - Sufficient data  
  - Sequence formatting

---

## 4. Train/Test Split (Time-based)

- Data was split using **80% past → 20% future**.  
- **No shuffling** was performed.  
- This preserves the natural **time order**, which is critical for time-series forecasting.

---

##  5. Random Forest Model

### ✔ Why Random Forest?

Random Forest works extremely well on:

- Structured **tabular data**  
- **Engineered features** (lags, rolling stats, calendar variables)  
- **Non-linear relationships**  
- **Small to medium-sized datasets**

### ✔ Results

In this project, **Random Forest performed better than LSTM**.

####  Why Random Forest beat LSTM here:

- **Dataset size is small**  
  - LSTMs generally need **large sequential datasets** to shine.

- **Feature engineering was strong**  
  - Lag features + rolling averages already encode temporal structure well.

- **Noise and irregularity**  
  - Tree models like Random Forest handle noisy patterns robustly.

- **LSTM underfit on this dataset**  
  - With limited data, LSTMs tend to output near-constant or smoothed predictions.

- **Random Forest does not require scaling**  
  - LSTM is more sensitive to feature scaling and sequence formatting.

### ✔ Interpretation

Random Forest captured the **non-linear relationships** between lag features, rolling means, and demand **more effectively** than LSTM in this particular setting.

---

##  6. LSTM Model (Deep Learning)

### ✔ Why LSTM?

LSTMs are designed to capture **temporal dependencies** by directly learning from input **sequences**.

### ✔ Sequence Construction

A sequence window of **14 days** was used:

- **Input:** 14-day window of features  
- **Target:** demand on **day 15**  

Each training sample corresponds to:

> “Given the last 14 days, predict demand for the next day.”

### ✔ Why LSTM performed worse here:

- Dataset was **too small** for deep learning to generalize well.  
- LSTM requires **more historical depth** and data density.  
- Harder to learn strong seasonality without large sequences.  
- LSTM is **sensitive to scaling and hyperparameters**.  
- With limited data, LSTMs often tend to output **“flat” or smoothed predictions**.

### ✔ Still Valuable for the Project

Including the LSTM model demonstrates:

- ✔ Knowledge of deep learning architectures  
- ✔ Understanding of **sequence modeling**  
- ✔ Ability to **compare classical ML vs DL** and choose the appropriate one

---

##  7. Dynamic Pricing Engine (Simulated)

Since the dataset does not contain a price column, a **simple price elasticity model** was used to simulate how demand responds to different prices.

### ✔ Price–Demand Formula

We assume a **linearized price elasticity** around a base price:

\[
d(p) = d_{\text{base}} \times \left( 1 - e \cdot \frac{p - p_{\text{base}}}{p_{\text{base}}} \right)
\]

Where:

- \( d(p) \) = simulated demand at price \( p \)  
- \( d_{\text{base}} \) = predicted demand at base price (from the model)  
- \( p_{\text{base}} = 100 \) → reference/base price  
- \( e \) = elasticity coefficient (e.g., 0.4)  

**Intuition:**

- If \( p > p_{\text{base}} \) → demand decreases  
- If \( p < p_{\text{base}} \) → demand increases  
- Larger \( e \) → more sensitive demand is to price changes

### ✔ Revenue Formula

\[
R(p) = p \times d(p)
\]

### ✔ What the Pricing Engine Does

1. Takes predicted demand (`d_base`) from the ML/DL model.  
2. Simulates demand over a range of candidate prices (e.g., ₹60–₹140).  
3. Computes **revenue** at each price using `R(p) = p * d(p)`.  
4. Selects the price with **maximum simulated revenue**.

### ✔ Why this is useful

This converts **pure forecasting** into a **business decision tool**.

Companies apply similar ideas in:

- Dynamic **surge pricing** (e.g., Uber)  
- **Hotel room** pricing  
- **Airline** ticket pricing  
- **E-commerce** promotional pricing  

---

## 📈 8. Visualizations

The notebook includes several helpful visualizations:

### ✔ Distribution of Target

- Helps understand **skewness**, **spread**, and **outliers** in daily demand.

### ✔ Correlation Heatmap

- Reveals relationships between features.  
- Highlights which features are most related to demand.

### ✔ Actual vs Predicted (Random Forest vs LSTM)

- Compares model behavior visually.  
- Shows that Random Forest tracks demand more closely than LSTM.

### ✔ Revenue vs Price Curve

- Plots **Revenue** as a function of **Price**.  
- Shows the approximate **optimal price** where revenue peaks.

---

##  9. Results Summary

| Model          | RMSE      | MAE       | R²        | Notes                       |
|----------------|----------:|----------:|----------:|-----------------------------|
| **Random Forest** | ⭐ Lower  | ⭐ Lower  | ⭐ Higher | Best-performing model       |
| **LSTM**          | Higher   | Higher   | Negative  | Underfit on small dataset   |

> **Key outcome:** On this dataset, **classical ML (Random Forest)** outperformed **deep learning (LSTM)**.

---

##  10. Key Insights

- **Classical ML beat Deep Learning** on this dataset  
  → Important reminder that **DL is not always better**.

- **Lag Features + Rolling Averages are powerful**  
  → Good feature engineering can make traditional models very strong.

- **LSTM requires larger sequential data** to truly shine  
  → Especially for capturing complex seasonality and long-term trends.

- Combining **demand forecasting + pricing** leads to an **end-to-end business solution**, not just a predictive model.

- A **revenue-maximizing price** can be computed by combining demand forecasts with a simple economic model.

---

##  11. Future Work

Potential improvements and extensions:

- Incorporate **real price data** instead of a simulated elasticity model.  
- Add external features:
  - Weather data  
  - Promotions/discount campaigns  
  - Holidays and events  
- Explore **attention-based models** (Transformers for time-series).  
- Perform **systematic hyperparameter tuning** for LSTM.  
- Add classical time-series baselines:
  - **Prophet**  
  - **ARIMA / SARIMA**  

---

##  Author

** Pranavi Devulapalli**  

