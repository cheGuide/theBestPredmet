import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
import seaborn as sns
import mlflow

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def accuracy(pred, y):
    return (pred == y).mean()

def even_odd_labels(y_digit):
    return (y_digit.astype(int) % 2).reshape(-1, 1)

def load_mnist_even_odd(compose_two_digits=False, test_size=0.2, seed=42):
    print("→ Загружаю MNIST (может занять ~1‑2 мин при первом запуске)…")
    mnist = fetch_openml('mnist_784', parser='auto')
    X = mnist['data'].to_numpy().astype(np.float32) / 255.0
    y_digits = mnist['target'].to_numpy().astype(int)

    if compose_two_digits:
        idx = np.random.permutation(len(X))
        X_left, X_right = X[idx[:len(X)//2]], X[idx[len(X)//2:]]
        y_left, y_right = y_digits[idx[:len(X)//2]], y_digits[idx[len(X)//2:]]
        X = np.hstack([X_left.reshape(-1, 28, 28), X_right.reshape(-1, 28, 28)]).reshape(-1, 28*56)
        y_digits = y_left * 10 + y_right
        print("  Сгенерировано двухзначных примеров:", len(X))

    y = even_odd_labels(y_digits)
    X_train, X_test, y_train, y_test, _, y_digits_test = train_test_split(
        X, y, y_digits, test_size=test_size, random_state=seed, stratify=y)

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, y_digits_test


class GradientDescent:
    def __init__(self, lr=0.1):
        self.lr = lr
    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]

class LogisticRegression:
    def __init__(self, n_features, lam=1e-4):
        self.theta = np.zeros((n_features, 1))
        self.lam = lam
    def forward(self, X):
        return sigmoid(X @ self.theta)
    def loss_grad(self, X, y):
        m = len(X)
        a = self.forward(X)
        grad = X.T @ (a - y) / m + self.lam * self.theta
        return grad
    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

class NeuralNetwork:
    def __init__(self, n_in, n_hidden, lam=1e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2/n_in), size=(n_in, n_hidden)),
            "b1": np.zeros((1, n_hidden)),
            "W2": rng.normal(0, np.sqrt(2/n_hidden), size=(n_hidden, 1)),
            "b2": np.zeros((1, 1))
        }
        self.lam = lam
    def _forward(self, X):
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = sigmoid(z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = sigmoid(z2)
        cache = (X, z1, a1, z2, a2)
        return a2, cache
    def _backward(self, cache, y):
        X, z1, a1, z2, a2 = cache
        m = len(X)
        dz2 = a2 - y
        dW2 = a1.T @ dz2 / m + self.lam * self.params["W2"]
        db2 = dz2.mean(axis=0, keepdims=True)
        dz1 = (dz2 @ self.params["W2"].T) * a1 * (1 - a1)
        dW1 = X.T @ dz1 / m + self.lam * self.params["W1"]
        db1 = dz1.mean(axis=0, keepdims=True)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    def grads(self, X, y):
        a2, cache = self._forward(X)
        return self._backward(cache, y)
    def predict(self, X, batch=1024):
        out = []
        for i in range(0, len(X), batch):
            a2, _ = self._forward(X[i:i+batch])
            out.append(a2)
        return (np.vstack(out) >= 0.5).astype(int)

def train(model, optimizer, X_train, y_train, epochs, batch, X_test, y_test):
    history = {"train_acc": [], "test_acc": []}
    idx = np.arange(len(X_train))

    with mlflow.start_run():
        mlflow.log_param("lr", optimizer.lr)
        mlflow.log_param("batch", batch)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("model", "logreg" if isinstance(model, LogisticRegression) else "nn")
        mlflow.log_param("lam", model.lam if hasattr(model, 'lam') else "n/a")

        for ep in range(1, epochs+1):
            np.random.shuffle(idx)
            for start in range(0, len(X_train), batch):
                b = idx[start:start+batch]
                if isinstance(model, LogisticRegression):
                    grad = model.loss_grad(X_train[b], y_train[b])
                    optimizer.step({"theta": model.theta}, {"theta": grad})
                else:
                    grads = model.grads(X_train[b], y_train[b])
                    optimizer.step(model.params, grads)
            if ep == 1 or ep % max(1, epochs//10) == 0 or ep == epochs:
                train_acc = accuracy(model.predict(X_train), y_train)
                test_acc  = accuracy(model.predict(X_test),  y_test)
                # 1. Loss (binary cross-entropy)
                eps = 1e-8
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                train_loss = -np.mean(
                    y_train * np.log(train_preds + eps) + (1 - y_train) * np.log(1 - train_preds + eps))
                test_loss = -np.mean(y_test * np.log(test_preds + eps) + (1 - y_test) * np.log(1 - test_preds + eps))

                # 2. Разность точности
                acc_gap = train_acc - test_acc

                # 3. Норма весов (регуляризация)
                if isinstance(model, LogisticRegression):
                    weight_norm = np.linalg.norm(model.theta)
                else:
                    weight_norm = sum(np.linalg.norm(p) for k, p in model.params.items() if 'W' in k)

                # 4. Средняя предсказанная вероятность (уверенность)
                train_conf = np.mean(train_preds)
                test_conf = np.mean(test_preds)

                # 🔹 Логируем всё в MLflow:
                mlflow.log_metric("train_loss", train_loss, step=ep)
                mlflow.log_metric("test_loss", test_loss, step=ep)
                mlflow.log_metric("acc_gap", acc_gap, step=ep)
                mlflow.log_metric("weight_norm", weight_norm, step=ep)
                mlflow.log_metric("train_conf", train_conf, step=ep)
                mlflow.log_metric("test_conf", test_conf, step=ep)

                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                print(f"Epoch {ep:3d}/{epochs}: train\u202f{train_acc:.3f}  test\u202f{test_acc:.3f}")
                mlflow.log_metric("train_acc", train_acc, step=ep)
                mlflow.log_metric("test_acc", test_acc, step=ep)

    return history
def plot_conf_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Чёт", "Неч"], yticklabels=["Чёт", "Неч"])
    plt.xlabel("Истинные метки")
    plt.ylabel("Предсказанные метки")
    plt.title("Матрица ошибок (2×2)")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"→ Матрица ошибок сохранена: {filename}")



def show_test_examples(model, X_test, y_test, num_examples=25):
    idx = np.random.choice(len(X_test), size=num_examples, replace=False)
    samples = X_test[idx]
    labels = y_test[idx].flatten()
    preds = model.predict(samples).flatten()
    img_shape = (28, 28) if samples.shape[1] == 784 else (28, 56)

    plt.figure(figsize=(12, 8))
    for i in range(num_examples):
        plt.subplot(5, 5, i + 1)
        plt.imshow(samples[i].reshape(img_shape), cmap='gray')
        title = f"Пред: {'Чёт' if preds[i]==0 else 'Неч'}\nИст: {'Чёт' if labels[i]==0 else 'Неч'}"
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(title, color=color, fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    print("→ Примеры предсказаний сохранены: sample_predictions.png")



def plot_conf_matrix_digits(y_pred, y_digits, filename="confusion_matrix_2x10.png"):
    matrix = np.zeros((2, 10), dtype=int)
    for pred, digit in zip(y_pred.flatten(), y_digits.flatten()):
        digit_class = int(digit) % 10
        matrix[int(pred), digit_class] += 1

    plt.figure(figsize=(8, 4))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=range(10), yticklabels=["Чёт", "Неч"])
    plt.xlabel("Последняя цифра")
    plt.ylabel("Предсказанная чётность")
    plt.title("Матрица ошибок (2×10)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"→ Матрица ошибок сохранена: {filename}")



def draw_and_predict(model):
    import cv2
    print("→ Нарисуйте цифру мышкой (белым по чёрному), нажмите [Esc] для предсказания")
    canvas = np.zeros((280, 280), dtype=np.uint8)
    drawing = False

    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(canvas, (x, y), 10, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Draw")
    cv2.setMouseCallback("Draw", draw)

    while True:
        cv2.imshow("Draw", canvas)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    img = cv2.resize(canvas, (28, 28))
    img = img.astype(np.float32) / 255.0
    vec = img.reshape(1, -1)
    pred = model.predict(vec)[0][0]
    print(f"→ Предсказание: {'Чётная' if pred == 0 else 'Нечётная'}")

def predict_from_image(model, filepath):
    import cv2
    if not os.path.exists(filepath):
        print(f"❌ Файл {filepath} не найден.")
        return
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Невозможно загрузить изображение: {filepath}")
        return
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (20, 20))
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = img
    img = canvas.astype(np.float32) / 255.0
    vec = img.reshape(1, -1)
    pred = model.predict(vec)[0][0]
    print(f"→ Предсказание по {filepath}: {'Чётная' if pred == 0 else 'Нечётная'}")

def main():
    p = argparse.ArgumentParser(description="Лабораторная: GD, LogReg, NN, MNIST even/odd")
    p.add_argument("--model", choices=["logreg", "nn"], default="nn")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--lr",     type=float, default=0.8)
    p.add_argument("--lam",    type=float, default=1e-4)
    p.add_argument("--compose",action="store_true")
    p.add_argument("--save",   metavar="FILE")
    p.add_argument("--load",   metavar="FILE")
    p.add_argument("--draw",   action="store_true")
    p.add_argument("--img",    metavar="FILE", help="загрузить изображение и распознать")
    args = p.parse_args()

    X_train, X_test, y_train, y_test, y_digits_test = load_mnist_even_odd(args.compose)


    if args.model == "logreg":
        model = LogisticRegression(X_train.shape[1], args.lam)
    else:
        model = NeuralNetwork(X_train.shape[1], args.hidden, args.lam)

    if args.load:
        data = np.load(args.load, allow_pickle=True)
        if args.model == "logreg":
            model.theta[:] = data["theta"]
        else:
            for k in model.params: model.params[k][:] = data[k]
        print("→ Веса загружены, точность:",
              accuracy(model.predict(X_test), y_test))

        if args.img:
            predict_from_image(model, args.img)
            sys.exit(0)
        if args.draw:
            draw_and_predict(model)
            sys.exit(0)
        show_test_examples(model, X_test, y_test)
        plot_conf_matrix_digits(model.predict(X_test), y_digits_test)

        sys.exit(0)

    if args.img:
        print("❗ Указан --img, но модель не загружена. Используйте также --load.")
        sys.exit(1)

    if args.draw:
        draw_and_predict(model)
        sys.exit(0)

    opt = GradientDescent(args.lr)
    train(model, opt, X_train, y_train, args.epochs, args.batch, X_test, y_test)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        if args.model == "logreg":
            np.savez(args.save, theta=model.theta)
        else:
            np.savez(args.save, **model.params)
        print("→ Веса сохранены в", args.save)

    show_test_examples(model, X_test, y_test)

if __name__ == "__main__":
    main()
