import json
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Настройки
DEBUG = False  # Включить/выключить отладочный вывод
PLOT = True  # Включить/выключить построение графика


# Функция загрузки правил из JSON-файла
def read_rules(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Функция загрузки фактов из JSON-файла
def read_facts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Функция обработки правил
def process_rules(rules, facts):
    known_facts = set(facts)
    new_facts = []

    if DEBUG:
        print("\n=== НАЧИНАЕМ ОБРАБОТКУ ПРАВИЛ ===\n")

    for rule in rules:
        if "if" not in rule or "then" not in rule:
            continue  # Пропускаем некорректные правила

        condition = rule["if"]
        operator = list(condition.keys())[0]
        values = set(condition[operator])
        outcome = rule["then"]

        if DEBUG:
            print(f" Проверяем правило: {rule} (известные факты: {sorted(known_facts)})")

        if operator == "and" and values.issubset(known_facts):
            new_facts.append(outcome)
        elif operator == "or" and values.intersection(known_facts):
            new_facts.append(outcome)
        elif operator == "not" and not values.intersection(known_facts) and outcome not in known_facts:
            new_facts.append(outcome)

    known_facts.update(new_facts)
    return list(known_facts)


# Функция градиентного спуска
def gradient_descent(X, y, learning_rate=1e-10, epochs=20000):
    m = len(y)  # Количество данных
    theta_0 = 0  # Начальное значение свободного члена
    theta_1 = 0  # Начальное значение коэффициента

    for _ in range(epochs):
        y_pred = theta_0 + theta_1 * X  # Предсказание модели
        error = y_pred - y  # Ошибка предсказания

        # Градиенты (частные производные)
        d_theta_0 = (1/m) * np.sum(error)
        d_theta_1 = (1/m) * np.sum(error * X)

        # Проверка на NaN и слишком большие значения
        if np.isnan(d_theta_0) or np.isnan(d_theta_1) or abs(d_theta_0) > 1e5 or abs(d_theta_1) > 1e5:
            print(" Градиентный спуск остановлен: слишком большие значения")
            break

        # Обновление параметров
        theta_0 -= learning_rate * d_theta_0
        theta_1 -= learning_rate * d_theta_1

    return theta_0, theta_1


# Оптимизированная версия тестирования производительности
def benchmark_execution():
    rule_sizes = np.linspace(500, 100000, num=100, dtype=int).tolist()
    times = []

    for size in rule_sizes:
        max_code = 500
        max_items = 5
        num_facts = 500
        rules = create_random_rules(max_code, max_items, size)
        facts = create_facts(max_code, num_facts)

        start = time.time()
        process_rules(rules, facts)
        end = time.time()

        times.append(end - start)
        print(f"Обработано {size} правил за {end - start:.6f} секунд")

    return rule_sizes, times


# Построение двух графиков
def visualize_performance():
    rule_sizes, times = benchmark_execution()

    X = np.array(rule_sizes).reshape(-1, 1)
    y = np.array(times)

    # Линейная регрессия (sklearn)
    model = LinearRegression()
    model.fit(X, y)
    trend_sklearn = model.predict(X)

    # Градиентный спуск
    theta_0, theta_1 = gradient_descent(X.flatten(), y)
    trend_gd = theta_0 + theta_1 * X.flatten()

    # Вывод коэффициентов
    print(f"\n Линейная регрессия: θ0 = {model.intercept_:.6f}, θ1 = {model.coef_[0]:.6f}")
    print(f" Градиентный спуск: θ0 = {theta_0:.6f}, θ1 = {theta_1:.6f}")

    # Создаём две оси (два графика в одном окне)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- ГРАФИК 1: ЛИНЕЙНАЯ РЕГРЕССИЯ ---
    axs[0].plot(rule_sizes, times, marker='o', label='Фактические данные')
    axs[0].plot(rule_sizes, trend_sklearn, linestyle='--', label='Линейная регрессия')
    axs[0].set_xlabel("Количество правил")
    axs[0].set_ylabel("Время выполнения (сек)")
    axs[0].set_title("Линейная регрессия")
    axs[0].legend()
    axs[0].grid()

    # --- ГРАФИК 2: ГРАДИЕНТНЫЙ СПУСК ---
    axs[1].plot(rule_sizes, times, marker='o', label='Фактические данные')
    axs[1].plot(rule_sizes, trend_gd, linestyle='-', color='green', label='Градиентный спуск')
    axs[1].set_xlabel("Количество правил")
    axs[1].set_ylabel("Время выполнения (сек)")
    axs[1].set_title("Градиентный спуск")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()


# Генерация случайных правил
def create_random_rules(max_code, max_items, num_rules, operators=["and", "or", "not"]):
    rule_set = []
    for _ in range(num_rules):
        operator = random.choice(operators)
        items = [random.randint(1, max_code) for _ in range(random.randint(2, max_items))]
        rule = {"if": {operator: items}, "then": max_code + _}
        rule_set.append(rule)
    return rule_set


# Генерация случайных фактов
def create_facts(max_code, num_facts):
    return [random.randint(1, max_code) for _ in range(num_facts)]


# Основной код программы
if __name__ == "__main__":
    use_json = input("Загрузить данные из JSON-файлов? (yes/no): ").strip().lower()

    if use_json == "yes":
        rule_data = read_rules("rules.json")
        fact_data = read_facts("facts.json")
    else:
        max_code = 500
        max_items = 5
        num_rules = 100000
        num_facts = 500
        rule_data = create_random_rules(max_code, max_items, num_rules)
        fact_data = create_facts(max_code, num_facts)

    final_result = process_rules(rule_data, fact_data)
    print("\n=== Финальные факты ===")
    print(sorted(final_result))

    if PLOT:
        visualize_performance()