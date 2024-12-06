import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Neural Network Implementation
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.weights = []
        self.biases = []
        self.a = []
        self.activations = {
            "sigmoid": (sigmoid, sigmoid_derivative),
            "tanh": (tanh, tanh_derivative),
            "relu": (relu, relu_derivative),
        }
        self.init_weights()

    def init_weights(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.biases.append(np.random.randn(layers[i + 1]) * 0.1)
            self.a.append(np.zeros((1, layers[i])))
        self.a.append(np.zeros((1, layers[-1])))

    def forward(self, x):
        self.a[0] = x
        activation_func = self.activations[self.activation][0]
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            self.a[i + 1] = activation_func(z)
        return self.a[-1]

    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        derivative_func = self.activations[self.activation][1]

        delta = self.a[-1] - y
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.a[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db.reshape(self.biases[i].shape)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * derivative_func(self.a[i])

    def train(self, x, y, epochs, learning_rate, update_callback=None):
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y, learning_rate)
            if update_callback:
                update_callback()
            if epoch % 100 == 0:
                loss = np.mean((self.a[-1] - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")


class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Neural Network Visualizer")
        self.create_ui()

    def create_ui(self):
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TEntry", font=("Helvetica", 10))

        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Dataset", command=self.load_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Number of Inputs:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.input_size = tk.IntVar(value=2)
        ttk.Entry(frame, textvariable=self.input_size).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Hidden Layers (comma-separated):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.hidden_layers = tk.StringVar(value="4,4")
        ttk.Entry(frame, textvariable=self.hidden_layers).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Number of Outputs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_size = tk.IntVar(value=1)
        ttk.Entry(frame, textvariable=self.output_size).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Activation Function:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.activation = tk.StringVar(value="sigmoid")
        ttk.Combobox(frame, textvariable=self.activation, values=["sigmoid", "tanh", "relu"]).grid(row=3, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Generate Network", command=self.generate_network).grid(row=4, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Load Dataset", command=self.load_dataset).grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Start Training", command=self.start_training_with_dataset).grid(row=4, column=2, padx=5, pady=5)

        self.progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=200, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, pady=10)

        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S))

    def show_about(self):
        messagebox.showinfo("About", "Enhanced Neural Network Visualizer\nDeveloped by Saugat Pahari")

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            data = pd.read_csv(file_path)

            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-7)

            if y.dtype == 'object':
                unique_classes = np.unique(y)
                class_mapping = {label: idx for idx, label in enumerate(unique_classes)}
                y = np.vectorize(class_mapping.get)(y)

            y = y.reshape(-1, 1)

            self.X = X
            self.y = y

            print("Dataset loaded successfully")

    def start_training_with_dataset(self):
        if not hasattr(self, 'network'):
            print("Network has not been initialized yet!")
            return
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            print("Dataset has not been loaded yet!")
            return

        def update_visualization():
            self.visualize_network()

        def train_step(epoch=0):
            if epoch < 1000:
                self.network.train(self.X, self.y, epochs=1, learning_rate=0.01, update_callback=update_visualization)
                self.visualize_network()
                self.root.after(10, train_step, epoch + 1)

        train_step()

    def generate_network(self):
        input_size = self.input_size.get()
        hidden_layers = list(map(int, self.hidden_layers.get().split(",")))
        output_size = self.output_size.get()
        activation = self.activation.get()
        self.network = NeuralNetwork(input_size, hidden_layers, output_size, activation)
        self.visualize_network()

    def visualize_network(self):
        self.canvas.delete("all")
        layers = [self.network.input_size] + self.network.hidden_layers + [self.network.output_size]
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        layer_gap = width / len(layers)

        layer_positions = []
        for i, nodes in enumerate(layers):
            x = layer_gap * i + layer_gap / 2
            y_spacing = height / (nodes + 1)
            layer_positions.append([(x, y_spacing * (j + 1)) for j in range(nodes)])

        sample_index = 0  # Index of the sample to visualize

        for layer_idx, layer in enumerate(layer_positions):
            for node_idx, (x, y) in enumerate(layer):
                activation = self.network.a[layer_idx][sample_index][node_idx] if layer_idx < len(self.network.a) else 0
                activation_normalized = (activation - np.min(activation)) / (np.ptp(activation) + 1e-7)
                activation_normalized = max(0, min(activation_normalized, 1))
                color = f"#{int(255 * (1 - activation_normalized)):02x}{int(255 * activation_normalized):02x}00"
                self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill=color, outline="black")

        for i in range(len(layer_positions) - 1):
            for start_idx, start in enumerate(layer_positions[i]):
                for end_idx, end in enumerate(layer_positions[i + 1]):
                    weight = self.network.weights[i][start_idx, end_idx]
                    thickness = max(1, int(abs(weight * 5)))
                    color = "blue" if weight > 0 else "red"
                    self.canvas.create_line(start[0], start[1], end[0], end[1], fill=color, width=thickness)

        self.canvas.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()