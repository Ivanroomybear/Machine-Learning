import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = {
    'area': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1100, 1215, 1954, 2142, 1347, 1864, 1724, 
             2500, 800, 2200, 1150, 1450, 1650, 1850, 1250, 950, 2100, 1550, 1350, 1750, 1950, 2300, 1050],
    'bedrooms': [3, 2, 4, 1, 3, 5, 3, 1, 2, 4, 2, 3, 2, 5, 
                 4, 1, 3, 2, 4, 3, 2, 1, 2, 5, 3, 2, 4, 3, 4, 2],
    'age': [10, 15, 12, 14, 16, 13, 11, 9, 12, 16, 20, 13, 11, 14, 
            5, 25, 8, 12, 18, 10, 7, 15, 22, 6, 14, 11, 9, 12, 5, 20],
    'category': ['Standart', 'Luxury', 'Affordable', 'Comfortable', 'Standart', 'Affordable', 'Luxury', 'Comfortable', 'Comfortable', 'Luxury', 'Standart', 'Affordable', 'Luxury', 'Standart',
                 'Luxury', 'Affordable', 'Luxury', 'Comfortable', 'Standart', 'Luxury', 'Luxury', 'Comfortable', 'Affordable', 'Luxury', 'Standart', 'Comfortable', 'Luxury', 'Luxury', 'Luxury', 'Standart']
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'age']]
Y = df[['category']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

model = DecisionTreeClassifier(random_state = 42)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(f"Прогнозируемые категории: {Y_pred}")

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Точность модели: {accuracy}")

plt.figure(figsize = (8, 6))
plot_tree(model, feature_names=['area', 'bedrooms', 'age'], class_names=model.classes_, filled=True)
plt.show()