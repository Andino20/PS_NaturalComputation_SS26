import csv
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
LEARNING_RATE = 0.01
BATCH_SIZE = 32

ACTIVATIONS = ["ReLU", "GeLU", "Sigmoid", "TanH", "Swish", "Identity"]

PROBLEMS = {
    "Easy": {
        "train": ROOT / "problems/easy/two_moons_train.csv",
        "test": ROOT / "problems/easy/two_moons_test.csv",
        "hidden_layers": [8],
        "epochs": 100,
    },
    "Medium": {
        "train": ROOT / "problems/medium/concentric_circles_train.csv",
        "test": ROOT / "problems/medium/concentric_circles_test.csv",
        "hidden_layers": [8, 8],
        "epochs": 150,
    },
    "Hard": {
        "train": ROOT / "problems/hard/crossing_spirals_train.csv",
        "test": ROOT / "problems/hard/crossing_spirals_test.csv",
        "hidden_layers": [16, 16],
        "epochs": 250,
    },
}


def load_dataset(path):
    with path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        feature_names = [name for name in reader.fieldnames if name != "label"]
        features = []
        labels = []

        for row in reader:
            features.append([float(row[name]) for name in feature_names])
            labels.append(float(row["label"]))

    return feature_names, features, labels


def standardize(train_features, test_features):
    feature_count = len(train_features[0])
    means = []
    stds = []

    for col in range(feature_count):
        values = [row[col] for row in train_features]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std > 0 else 1.0)

    def transform(features):
        return [
            [(row[col] - means[col]) / stds[col] for col in range(feature_count)]
            for row in features
        ]

    return transform(train_features), transform(test_features)


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def activate_value(x, activation):
    if activation == "ReLU":
        return x if x > 0 else 0.0
    if activation == "GeLU":
        return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    if activation == "Sigmoid":
        return sigmoid(x)
    if activation == "TanH":
        return math.tanh(x)
    if activation == "Swish":
        return x * sigmoid(x)
    if activation == "Identity":
        return x

    raise ValueError(f"Unknown activation: {activation}")


def activation_derivative_value(x, activation):
    if activation == "ReLU":
        return 1.0 if x > 0 else 0.0
    if activation == "GeLU":
        normal_cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        normal_pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return normal_cdf + x * normal_pdf
    if activation == "Sigmoid":
        s = sigmoid(x)
        return s * (1.0 - s)
    if activation == "TanH":
        t = math.tanh(x)
        return 1.0 - t * t
    if activation == "Swish":
        s = sigmoid(x)
        return s + x * s * (1.0 - s)
    if activation == "Identity":
        return 1.0

    raise ValueError(f"Unknown activation: {activation}")


def matmul_add_bias(matrix, weights, bias):
    output = []

    for row in matrix:
        out_row = []
        for out_col in range(len(bias)):
            value = bias[out_col]
            for in_col, row_value in enumerate(row):
                value += row_value * weights[in_col][out_col]
            out_row.append(value)
        output.append(out_row)

    return output


def apply_activation(matrix, activation):
    return [[activate_value(value, activation) for value in row] for row in matrix]


def apply_sigmoid(matrix):
    return [[sigmoid(value) for value in row] for row in matrix]


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matmul(left, right):
    rows = len(left)
    inner = len(right)
    cols = len(right[0])
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for row in range(rows):
        for mid in range(inner):
            left_value = left[row][mid]
            for col in range(cols):
                result[row][col] += left_value * right[mid][col]

    return result


def zeros_like(matrix):
    return [[0.0 for _ in row] for row in matrix]


class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_activation, rng):
        self.hidden_activation = hidden_activation
        self.weights = []
        self.biases = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            layer_weights = [
                [rng.uniform(-limit, limit) for _ in range(fan_out)]
                for _ in range(fan_in)
            ]
            layer_biases = [0.0 for _ in range(fan_out)]
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)

    def forward(self, features):
        activations = [features]
        pre_activations = []
        current = features

        for layer_index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            z = matmul_add_bias(current, weights, bias)
            pre_activations.append(z)

            if layer_index == len(self.weights) - 1:
                current = apply_sigmoid(z)
            else:
                current = apply_activation(z, self.hidden_activation)

            activations.append(current)

        return activations, pre_activations

    def train_batch(self, features, labels):
        batch_size = len(features)
        activations, pre_activations = self.forward(features)
        predictions = activations[-1]

        delta = [
            [predictions[row][0] - labels[row]]
            for row in range(batch_size)
        ]

        weight_gradients = [None for _ in self.weights]
        bias_gradients = [None for _ in self.biases]

        for layer_index in reversed(range(len(self.weights))):
            previous_activation = activations[layer_index]
            previous_transposed = transpose(previous_activation)
            grad_w = matmul(previous_transposed, delta)
            grad_b = [0.0 for _ in self.biases[layer_index]]

            for row in delta:
                for col, value in enumerate(row):
                    grad_b[col] += value

            for row in range(len(grad_w)):
                for col in range(len(grad_w[row])):
                    grad_w[row][col] /= batch_size

            for col in range(len(grad_b)):
                grad_b[col] /= batch_size

            weight_gradients[layer_index] = grad_w
            bias_gradients[layer_index] = grad_b

            if layer_index > 0:
                weights_transposed = transpose(self.weights[layer_index])
                previous_delta = matmul(delta, weights_transposed)
                z_previous = pre_activations[layer_index - 1]
                adjusted_delta = zeros_like(previous_delta)

                for row in range(len(previous_delta)):
                    for col in range(len(previous_delta[row])):
                        adjusted_delta[row][col] = (
                            previous_delta[row][col]
                            * activation_derivative_value(
                                z_previous[row][col],
                                self.hidden_activation,
                            )
                        )

                delta = adjusted_delta

        for layer_index in range(len(self.weights)):
            for row in range(len(self.weights[layer_index])):
                for col in range(len(self.weights[layer_index][row])):
                    self.weights[layer_index][row][col] -= (
                        LEARNING_RATE * weight_gradients[layer_index][row][col]
                    )

            for col in range(len(self.biases[layer_index])):
                self.biases[layer_index][col] -= LEARNING_RATE * bias_gradients[layer_index][col]

    def predict_probabilities(self, features):
        activations, _ = self.forward(features)
        return [row[0] for row in activations[-1]]


def binary_cross_entropy(predictions, labels):
    epsilon = 1e-12
    loss = 0.0

    for prediction, label in zip(predictions, labels):
        clipped = min(max(prediction, epsilon), 1.0 - epsilon)
        loss += -(label * math.log(clipped) + (1.0 - label) * math.log(1.0 - clipped))

    return loss / len(labels)


def accuracy(predictions, labels):
    correct = 0

    for prediction, label in zip(predictions, labels):
        predicted_label = 1.0 if prediction >= 0.5 else 0.0
        if predicted_label == label:
            correct += 1

    return correct / len(labels)


def train_model(problem_name, problem_config, activation):
    feature_names, train_features, train_labels = load_dataset(problem_config["train"])
    test_feature_names, test_features, test_labels = load_dataset(problem_config["test"])

    if feature_names != test_feature_names:
        raise ValueError(f"Train/test feature mismatch for {problem_name}")

    train_features, test_features = standardize(train_features, test_features)
    layer_sizes = [len(feature_names)] + problem_config["hidden_layers"] + [1]
    rng = random.Random(RANDOM_SEED)
    model = NeuralNetwork(layer_sizes, activation, rng)

    indices = list(range(len(train_features)))
    for _ in range(problem_config["epochs"]):
        rng.shuffle(indices)

        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[batch_start : batch_start + BATCH_SIZE]
            batch_features = [train_features[index] for index in batch_indices]
            batch_labels = [train_labels[index] for index in batch_indices]
            model.train_batch(batch_features, batch_labels)

    train_predictions = model.predict_probabilities(train_features)
    test_predictions = model.predict_probabilities(test_features)

    return {
        "problem": problem_name,
        "activation": activation,
        "topology": "-".join(str(size) for size in layer_sizes),
        "epochs": problem_config["epochs"],
        "train_loss": binary_cross_entropy(train_predictions, train_labels),
        "test_loss": binary_cross_entropy(test_predictions, test_labels),
        "train_accuracy": accuracy(train_predictions, train_labels),
        "test_accuracy": accuracy(test_predictions, test_labels),
    }


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "activation",
        "topology",
        "epochs",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
    ]

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        for row in rows:
            formatted = dict(row)
            for key in [
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy",
            ]:
                formatted[key] = f"{row[key]:.6f}"
            writer.writerow(formatted)


def main():
    rows = []

    for problem_name, problem_config in PROBLEMS.items():
        for activation in ACTIVATIONS:
            print(f"Training {problem_name} with {activation}...")
            rows.append(train_model(problem_name, problem_config, activation))

    output_path = ROOT / "results/baseline_results.csv"
    write_results(output_path, rows)
    print(f"Results written to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
