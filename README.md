# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 10-10-2025

#### NAME: Sarish Varshan V
#### REGISTER NUMBER: 212223230196

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.

### PROGRAM
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

data = pd.read_csv("gold.csv")

data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data = data.sort_values('Date')

data['Price'] = data['Price'].replace(',', '', regex=True).astype(float)
data = data.set_index('Date')

ts = data['Price']

result = adfuller(ts.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

plot_acf(ts.dropna(), lags=30)
plot_pacf(ts.dropna(), lags=30)
plt.show()

train_size = int(len(ts) * 0.8)
train, test = ts[0:train_size], ts[train_size:]

lags = min(5, len(train) - 1)   # Ensure lags < training size
model = AutoReg(train, lags=lags).fit()
print(model.summary())

preds = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
preds.index = test.index  # Align dates properly

error = mean_squared_error(test, preds)
print("MSE:", round(error, 2))

plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, preds, label='Predicted', color='red')
plt.title("Gold Price Forecasting (AutoReg Model)")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")
plt.legend()
plt.show()

```
### OUTPUT:

### PACF - ACF
<img width="709" height="542" alt="image" src="https://github.com/user-attachments/assets/51fbf5a2-41e6-4672-bd17-90586a10688a" />

<img width="705" height="541" alt="image" src="https://github.com/user-attachments/assets/c0294861-8012-43e6-8093-e9cdaf534eff" />

### FINIAL PREDICTION
<img width="1075" height="584" alt="image" src="https://github.com/user-attachments/assets/34deb045-6851-46d9-b5e2-ceb3b0ef86c0" />



### RESULT:
Thus we have successfully implemented the auto regression function using python.
