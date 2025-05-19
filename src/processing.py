import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

root = Path(__file__).parent.parent / "data"

# https://www.kaggle.com/datasets/timmofeyy/real-estate-in-moscow-for-sale-20220311
data = pd.read_csv(root / "raw" / "moscow_real_estate_sale.csv")

# 1. Первичный анализ данных
print("=== Основная информация о датасете ===")
print(f"Количество строк: {len(data)}")
print(f"Количество столбцов: {len(data.columns)}")
print("\nПервые 3 строки данных:")
print(data.head(3))
print("\nИнформация о датасете:")
print(data.info())
print("\nОписательная статистика:")
print(data.describe())

# 2. Выбор 5 указанных столбцов
selected_columns = ["total_area", "rooms", "storeys", "storey", "price"]
selected_data = data[selected_columns]

print("\n=== Выбранные столбцы ===")
print(selected_data.head())

selected_data.to_csv(root / "processed" / "all.csv", index=False)
selected_data = selected_data[selected_data["rooms"] != "+"]


# Разбиение на train/test (80%/20%) с фиксированным random_state для воспроизводимости
df_train, df_test = train_test_split(selected_data, test_size=0.2, random_state=42)

# Сохранение train и test наборов
df_train.to_csv(root / "processed" / "train.csv", index=False)
df_test.to_csv(root / "processed" / "test.csv", index=False)

print("\n=== Размеры выборок ===")
print(f"Обучающая выборка: {len(df_train)} записей")
print(f"Тестовая выборка: {len(df_test)} записей")
