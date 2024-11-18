import math
import random
import tkinter as tk
from tkinter import StringVar, IntVar
import logging
from AppKit import NSOpenPanel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
learning_rate = 0.01
weight_penalty = 0.01  # L2 Regularization

# Xavier initialization
def xavier_init(input_size, output_size):
    return random.uniform(-math.sqrt(6 / (input_size + output_size)), math.sqrt(6 / (input_size + output_size)))

# Activation functions and derivatives
def sigmoid(x):
    try:
        if x > 20:
            return 1.0
        elif x < -20:
            return 0.0
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        logging.error(f"Overflow in sigmoid with x={x}")
        return 0.0 if x < 0 else 1.0

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - x**2

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

# Neuron class
class Neuron:
    def __init__(self, activation_func='sigmoid', input_idx=-1):
        self.inputs = []
        self.bias = random.uniform(-1, 1)
        self.result = 0.0
        self.error = 0.0
        self.index = input_idx
        self.activation_func = activation_func

    def activate(self, total):
        if self.activation_func == 'sigmoid':
            return sigmoid(total)
        elif self.activation_func == 'tanh':
            return tanh(total)
        elif self.activation_func == 'relu':
            return relu(total)

    def activate_derivative(self, result):
        if self.activation_func == 'sigmoid':
            return sigmoid_derivative(result)
        elif self.activation_func == 'tanh':
            return tanh_derivative(result)
        elif self.activation_func == 'relu':
            return relu_derivative(result)

    def forward_prop(self, inputs):
        try:
            if self.index >= 0:
                self.result = inputs[self.index]
            else:
                total = sum(n.result * w for n, w in self.inputs) + self.bias
                if abs(total) > 50:
                    logging.warning(f"Extreme value in forward_prop: total={total}")
                self.result = self.activate(total)
        except Exception as e:
            logging.error(f"Error during forward propagation: {e}")
            raise

    def back_prop(self, learning_rate):
        if self.inputs:
            gradient = self.activate_derivative(self.result)
            for n, w in self.inputs:
                n.error += self.error * w
                l2_term = weight_penalty * w
                new_weight = w - learning_rate * (self.error * gradient * n.result + l2_term)
                self.inputs[self.inputs.index((n, w))] = (n, new_weight)
            self.bias -= learning_rate * gradient * self.error

# Neural Network class
class Network:
    def __init__(self, num_inputs, num_outputs, hidden_layers, hidden_width, activation_func='sigmoid'):
        self.inputs = [Neuron(input_idx=i) for i in range(num_inputs)]
        self.hidden_layers = [[Neuron(activation_func=activation_func) for _ in range(hidden_width)]
                              for _ in range(hidden_layers)]
        self.outputs = [Neuron(activation_func=activation_func) for _ in range(num_outputs)]
        self.connect_layers()

    def connect_layers(self):
        for idx, layer in enumerate(self.hidden_layers):
            source = self.inputs if idx == 0 else self.hidden_layers[idx - 1]
            for neuron in layer:
                for src_neuron in source:
                    neuron.inputs.append((src_neuron, xavier_init(len(source), len(layer))))
        for out_neuron in self.outputs:
            for src_neuron in self.hidden_layers[-1]:
                out_neuron.inputs.append((src_neuron, xavier_init(len(self.hidden_layers[-1]), len(self.outputs))))

    def forward_prop(self, inputs):
        for neuron in self.inputs:
            neuron.forward_prop(inputs)
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.forward_prop(inputs)
        for neuron in self.outputs:
            neuron.forward_prop(inputs)

    def back_prop(self, targets):
        for idx, out_neuron in enumerate(self.outputs):
            out_neuron.error = targets[idx] - out_neuron.result
        for layer in reversed(self.hidden_layers):
            for neuron in layer:
                neuron.back_prop(learning_rate)

# Neural Network UI class
class NNConfigUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neural Network Configuration")
        self.geometry("1024x768")
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Activation Function:").grid(row=0, column=0, sticky="w")
        self.activation_choice = StringVar(value="sigmoid")
        tk.OptionMenu(self, self.activation_choice, "sigmoid", "tanh", "relu").grid(row=0, column=1, sticky="w")

        tk.Label(self, text="Number of Hidden Layers:").grid(row=1, column=0, sticky="w")
        self.num_hidden_layers = IntVar(value=2)
        tk.Entry(self, textvariable=self.num_hidden_layers).grid(row=1, column=1, sticky="w")

        tk.Label(self, text="Width of Hidden Layers:").grid(row=2, column=0, sticky="w")
        self.layer_width = IntVar(value=5)
        tk.Entry(self, textvariable=self.layer_width).grid(row=2, column=1, sticky="w")

        tk.Label(self, text="Number of Inputs:").grid(row=3, column=0, sticky="w")
        self.num_inputs = IntVar(value=3)
        tk.Entry(self, textvariable=self.num_inputs).grid(row=3, column=1, sticky="w")

        tk.Label(self, text="Number of Outputs:").grid(row=4, column=0, sticky="w")
        self.num_outputs = IntVar(value=1)
        tk.Entry(self, textvariable=self.num_outputs).grid(row=4, column=1, sticky="w")

        tk.Button(self, text="Load Dataset", command=self.load_dataset).grid(row=5, column=0, sticky="w")
        self.dataset_path = StringVar(value="No file selected")
        tk.Label(self, textvariable=self.dataset_path).grid(row=5, column=1, sticky="w")

        tk.Button(self, text="Generate Network", command=self.generate_network).grid(row=6, column=0, sticky="w")
        tk.Button(self, text="Start Training", command=self.start_training).grid(row=7, column=0, sticky="w")

    def load_dataset(self):
        try:
            panel = NSOpenPanel.openPanel()
            panel.setCanChooseFiles_(True)
            panel.setCanChooseDirectories_(False)
            panel.setAllowsMultipleSelection_(False)
            panel.setAllowedFileTypes_(["csv"])
            if panel.runModal() == 1:
                file_path = str(panel.URLs()[0].path())
                self.dataset = pd.read_csv(file_path)
                self.numeric_data = self.dataset.select_dtypes(include=["number"])
                self.dataset_path.set(file_path)
                logging.info(f"Dataset loaded from {file_path}. Numeric columns: {list(self.numeric_data.columns)}")
            else:
                logging.warning("No file selected.")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")

    def generate_network(self):
        if not hasattr(self, 'numeric_data'):
            logging.error("Dataset not loaded. Please load the dataset before generating the network.")
            return

        target_col = self.numeric_data.columns[-1]
        self.num_inputs.set(len(self.numeric_data.columns) - 1)
        self.num_outputs.set(1 if self.numeric_data[target_col].nunique() > 2 else 2)

        self.network = Network(
            self.num_inputs.get(),
            self.num_outputs.get(),
            self.num_hidden_layers.get(),
            self.layer_width.get(),
            self.activation_choice.get()
        )
        logging.info("Neural network successfully generated.")

    def start_training(self):
        if not hasattr(self, 'numeric_data') or not hasattr(self, 'network'):
            logging.error("Dataset or network not initialized. Please load the dataset and generate the network first.")
            return

        target_col = self.numeric_data.columns[-1]
        inputs = self.numeric_data.drop(target_col, axis=1).values
        targets = self.numeric_data[target_col].values

        epochs = 50
        for epoch in range(epochs):
            for i, input_row in enumerate(inputs):
                target_row = [targets[i]]
                self.network.forward_prop(input_row)
                self.network.back_prop(target_row)
            logging.info(f"Epoch {epoch + 1}/{epochs} completed.")


if __name__ == "__main__":
    app = NNConfigUI()
    app.mainloop()
