import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Créer des données d'exemple
data = {
    'Age': [25, 30, 22, 40, 35, 50, 28, 60, 45, 33],
    'Taille': [170, 180, 160, 165, 175, 168, 172, 180, 169, 176],
    'Poids': [65, 75, 55, 68, 70, 80, 66, 85, 72, 74]
}
df = pd.DataFrame(data)

# 2. Séparer les caractéristiques (features) et la cible (target)
X = df[['Age', 'Taille']]  # Caractéristiques : Age, Taille
y = df['Poids']            # Cible : Poids

# 3. Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# 6. Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erreur quadratique moyenne (MSE) : {mse}")
print(f"Score R² : {r2}")

# 7. Visualisation des résultats
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='dotted')
plt.xlabel('Poids réel')
plt.ylabel('Poids prédit')
plt.title('Prédictions vs Réels')
plt.show()

# Exemple de nouvelles données
new_data = pd.DataFrame({
    'Age': [26, 38],
    'Taille': [172, 180]
})

# Faire des prédictions avec les nouvelles données
predictions = model.predict(new_data)
print(f"Prédictions du poids pour les nouvelles données : {predictions}")
