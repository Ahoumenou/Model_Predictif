import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as fg

from sklearn.preprocessing import MinMaxScaler


# preparation des données
df = pd.read_csv('advertising.csv')
df.head()
df.describe()

# df.corr()
# - tv     0.78
# - radio    0.58
# - journaux  0.23

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

from sklearn.model_selection import train_test_split
X = df[['tv','radio','journaux']]
y = df.ventes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg.fit(X_train, y_train)

y_pred_test = reg.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
print(f"RMSE: {mean_squared_error(y_test, y_pred_test)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_test)}")

df['tv2'] = df.tv**2
df['tvR'] = df.tv * df.radio
scaler = MinMaxScaler()
scaler.fit(df)

data_array = scaler.transform(df)

df = pd.DataFrame(data_array, columns = ['tv','radio','journaux','ventes','tv2','tvR'])

X = df[['tv','radio','journaux', 'tv2', 'tvR']]
y = df.ventes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg.fit(X_train, y_train)
y_hat_test = reg.predict(X_test)

print(f"Coefficients: {reg.coef_}")
print(f"RMSE: {mean_squared_error(y_test, y_hat_test)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_hat_test)}")




fg.scatter(y_test, y_pred_test)
fg.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
fg.title('Prédictions de ventes')
fg.xlabel('les ventes réelles')
fg.ylabel('les ventes predictes')
fg.show()



