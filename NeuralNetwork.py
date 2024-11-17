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
learning_rate = 0.1


# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Neuron class
class Neuron:
    def __init__(self, activation_func='sigmoid', input_idx=-1):
        self.inputs = []  # List of (Neuron, float) tuples
        self.bias = random.uniform(-1, 1)
        self.result = 0.0
        self.error = 0.0
        self.index = input_idx
        self.activation_func = activation_func

    def activate(self, total):
        if self.activation_func == 'sigmoid':
            return sigmoid(total)

    def activate_derivative(self, result):
        if self.activation_func == 'sigmoid':
            return sigmoid_derivative(result)

    def forward_prop(self, inputs):
        if self.index >= 0:
            self.result = inputs[self.index]
        else:
            total = sum(n.result * w for n, w in self.inputs) + self.bias
            self.result = self.activate(total)

    def back_prop(self, learning_rate):
        gradient = self.activate_derivative(self.result)
        for n, w in self.inputs:
            n.error += self.error * w
            new_weight = w - learning_rate * self.error * gradient * n.result
            self.inputs[self.inputs.index((n, w))] = (n, new_weight)
        self.bias -= learning_rate * self.error * gradient


# Neural Network
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
                    neuron.inputs.append((src_neuron, random.uniform(-1, 1)))
        for out_neuron in self.outputs:
            for src_neuron in self.hidden_layers[-1]:
                out_neuron.inputs.append((src_neuron, random.uniform(-1, 1)))

    def forward_prop(self, inputs):
        for layer in [self.inputs] + self.hidden_layers:
            for neuron in layer:
                neuron.forward_prop(inputs)
        for out_neuron in self.outputs:
            out_neuron.forward_prop(inputs)

    def back_prop(self, targets):
        for idx, out_neuron in enumerate(self.outputs):
            out_neuron.error = targets[idx] - out_neuron.result
        for layer in reversed(self.hidden_layers):
            for neuron in layer:
                neuron.back_prop(learning_rate)


# UI Class
class NNConfigUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neural Network Configuration")
        self.geometry("1024x768")

        # UI Elements
        tk.Label(self, text="Activation Function:").grid(row=0, column=0, sticky="w")
        self.activation_choice = StringVar(value="sigmoid")
        tk.OptionMenu(self, self.activation_choice, "sigmoid").grid(row=0, column=1, sticky="w")

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
        self.num_outputs = IntVar(value=2)
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

                # Automatically select numeric columns
                self.numeric_data = self.dataset.select_dtypes(include=["number"])
                self.dataset_path.set(file_path)
                logging.info(f"Dataset loaded from {file_path}. Numeric columns: {list(self.numeric_data.columns)}")
            else:
                logging.warning("No file selected.")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")

    def generate_network(self):
        self.network = Network(
            self.num_inputs.get(),
            self.num_outputs.get(),
            self.num_hidden_layers.get(),
            self.layer_width.get(),
            self.activation_choice.get()
        )
        logging.info("Neural network generated.")

    def start_training(self):
        if not hasattr(self, 'numeric_data'):
            logging.error("No dataset loaded or no numeric columns available.")
            return
        if not hasattr(self, 'network'):
            logging.error("Network not generated.")
            return

        accuracy = []
        epochs = 50

    # Ensure the last column is used as the target
        target_col = self.numeric_data.columns[-1]
        logging.info(f"Using '{target_col}' as the target column.")

        for epoch in range(epochs):
            correct_predictions = 0

            for _, row in self.numeric_data.iterrows():
                inputs = row[:-1].values  # Use all columns except the last one as inputs
                targets = [row.iloc[-1]]  # Ensure the target is treated as a list

                if len(targets) != len(self.network.outputs):
                    logging.error(f"Mismatch between targets ({len(targets)}) and output neurons ({len(self.network.outputs)}).")
                    return

                self.network.forward_prop(inputs)
                self.network.back_prop(targets)

            # Compare predictions with targets
                predicted = [1 if out.result >= 0.5 else 0 for out in self.network.outputs]
                correct_predictions += int(predicted == targets)

        # Calculate and log accuracy for this epoch
            epoch_accuracy = correct_predictions / len(self.numeric_data)
            accuracy.append(epoch_accuracy)
            logging.info(f"Epoch {epoch + 1}/{epochs}: Accuracy = {epoch_accuracy:.2f}")

        self.show_graph(accuracy)


    def show_graph(self, accuracy):
        fig, ax = plt.subplots()
        ax.plot(range(1, len(accuracy) + 1), accuracy, marker='o', label='Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy Over Epochs')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=8, column=0, columnspan=2)
        canvas.draw()


if __name__ == "__main__":
    app = NNConfigUI()
    app.mainloop()
