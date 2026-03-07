# Импорты
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Читаем данные из файла
df = pd.read_csv('house_prices.csv')

# Это статистика короче
print("Статистика цен")
avg_prices = df.groupby('category')['price'].mean().round(0)
print(avg_prices)

counts = df['category'].value_counts()
print(f"\nКатегории")
print(f"Больше всего: {counts.idxmax()} ({counts.max()} шт.)")
print(f"Меньше всего: {counts.idxmin()} ({counts.min()} шт.)")

# Важная вещь
X = df[['area', 'bedrooms', 'age']]
Y = df['category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, Y_train)

print(f"\nТочность: {accuracy_score(Y_test, model.predict(X_test))}")

# Решения программы
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, fontsize=8)
plt.show()
