import math
import random
import tkinter as tk
from tkinter import filedialog

num_inputs = 10
num_hidden_layers = 3
hidden_layer_width = 12
num_outputs = 4
neuron_scale = 5
axon_scale = 4
learning_rate = 0.1

training_data_size = 100


# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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



class Neuron:
    def __init__(self, x, y, input_idx=-1, bias=0.0, activation_func='sigmoid'):
        self.x = x
        self.y = y
        self.inputs = []
        self.outputs = []
        self.index = input_idx
        self.bias = bias
        self.result = 0.0
        self.error = 0.0
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


    def connect_input(self, in_n):
        in_axon = Axon(in_n, self)
        self.inputs.append(in_axon)

    def connect_output(self, out_n):
        out_axon = Axon(self, out_n)
        self.outputs.append(out_axon)

    def forward_prop(self, inputs):
        if self.result != 0.0:
            return self.result
        total = 0.0
        if self.index >= 0:
            total = inputs[self.index]
            #print(f'Input neuron {self.index} found value {inputs[self.index]}')
        else:
            for in_axon in self.inputs:
                in_n = in_axon.input
                in_n.forward_prop(inputs)
                in_val = in_n.result * in_axon.weight
                #print(f'Adding weighted value {in_val} from input')
                total += in_val
            #print(f'Neuron computed sum {total} from inputs')
        total += self.bias
        #print(f'Biased total is {total}')
        self.result = sigmoid(total)
        #print(f'Final neuron output is {self.result}')
        #print()

    def back_prop(self):
        if self.error != 0.0:
            return
        for out_axon in self.outputs:
            out_n = out_axon.output
            out_n.back_prop()
        gradient = self.result * (1.0 - self.result)
        delta = self.error * gradient
        # print(f'Error calc: {self.error} {gradient} {delta}')
        if self.index == -1:
            for in_axon in self.inputs:
                in_n = in_axon.input
                in_n.error += in_n.error * in_axon.weight
                in_axon.weight -= delta * in_n.result * learning_rate
        self.bias -= delta * learning_rate

    def draw(self, canvas, color='black'):
        canvas.create_oval(self.x - neuron_scale, self.y - neuron_scale, self.x + neuron_scale, self.y + neuron_scale, fill=color)


class Axon:
    def __init__(self, in_n, out_n, weight=0.0):
        self.input = in_n
        self.output = out_n
        self.weight = weight

    def draw(self, canvas, color='grey'):
        canvas.create_line(self.input.x,
                           self.input.y,
                           self.output.x,
                           self.output.y,
                           fill=color,
                           width=axon_scale)


class Network:
    def __init__(self):
        self.inputs = []
        self.hidden_layers = []
        self.outputs = []
        for idx in range(num_inputs):
            in_n = Neuron(0, 0, idx, 1.0)
            self.inputs.append(in_n)
        for layer in range(num_hidden_layers):
            self.hidden_layers.append([])
            for _ in range(hidden_layer_width):
                hidden_n = Neuron(0, 0)
                self.hidden_layers[layer].append(hidden_n)
                if layer == 0:
                    for in_n in self.inputs:
                        hidden_n.connect_input(in_n)
                        in_n.connect_output(hidden_n)
                else:
                    for h_n in self.hidden_layers[layer-1]:
                        hidden_n.connect_input(h_n)
                        h_n.connect_output(hidden_n)
        for _ in range(num_outputs):
            out_n = Neuron(0, 0)
            self.outputs.append(out_n)
            for h_n in self.hidden_layers[num_hidden_layers-1]:
                out_n.connect_input(h_n)
                h_n.connect_output(out_n)

    # Used in both train() and test() to get output from the current network
    def forward_prop(self, inputs):
        print(f'Performing forward propagation')
        for in_n in self.inputs:
            in_n.result = 0.0
        for h_layer in self.hidden_layers:
            for h_n in h_layer:
                h_n.result = 0.0
        for out_n in self.outputs:
            out_n.result = 0.0
            out_n.forward_prop(inputs)

    # Used with train() to modify weights and biases to reduce error
    def back_prop(self):
        print(f'Performing backward propagation')
        for h_layer in self.hidden_layers:
            for h_n in h_layer:
                h_n.error = 0.0
        for out_n in self.outputs:
            out_n.error = 0.0
        for in_n in self.inputs:
            in_n.error = 0.0
            in_n.back_prop()

    # Given a datum, produce an output and compare against the target, then perform backprop to reduce error
    # returns: nothing
    def train(self, data):
        self.forward_prop(data.inputs)
        for x in range(num_outputs):
            self.outputs[x].error = data.outputs[x] - self.outputs[x].result
            print(f'Output {x} error is {self.outputs[x].error}')
        self.back_prop()

    # Given a datum, produce an output
    # returns: the network output/selection/prediction
    def test(self, data):
        self.forward_prop(data.inputs)
        out_max = 0.0
        for out_n in self.outputs:
            out_max = max(out_max, out_n.result)
        return out_max


class RandData:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def generate(self):
        for _ in range(num_inputs):
            self.inputs.append(random.random()*10.0-5.0)
        for _ in range(num_outputs):
            self.outputs.append(random.random())


    def train():
        global training_data_size
        network = Network()
        for _ in range(training_data_size):
            dat = RandData()
            dat.generate()
            network.train(dat)

class NNConfigUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neural Network Configuration")
        self.geometry("800x600")
        
        # Neural network configuration options
        tk.Label(self, text="Activation Function:").grid(row=0, column=0, sticky="w")
        self.activation_choice = tk.StringVar(value="sigmoid")
        tk.OptionMenu(self, self.activation_choice, "sigmoid", "tanh", "relu").grid(row=0, column=1, sticky="w")
        
        tk.Label(self, text="Number of Hidden Layers:").grid(row=1, column=0, sticky="w")
        self.num_hidden_layers = tk.IntVar(value=3)
        tk.Entry(self, textvariable=self.num_hidden_layers).grid(row=1, column=1, sticky="w")

        tk.Label(self, text="Width of Hidden Layers:").grid(row=2, column=0, sticky="w")
        self.layer_width = tk.IntVar(value=12)
        tk.Entry(self, textvariable=self.layer_width).grid(row=2, column=1, sticky="w")

        # Dataset loading
        tk.Button(self, text="Load Dataset", command=self.load_dataset).grid(row=3, column=0, sticky="w")
        self.dataset_path = tk.StringVar(value="No file selected")
        tk.Label(self, textvariable=self.dataset_path).grid(row=3, column=1, sticky="w")

        # Run training
        tk.Button(self, text="Start Training", command=self.start_training).grid(row=4, column=0, sticky="w")

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xls;*.xlsx")])
        if file_path:
            self.dataset_path.set(file_path)

    def start_training(self):
        # Placeholder: replace with actual training logic
        print("Training started with settings:")
        print(f"Activation: {self.activation_choice.get()}")
        print(f"Hidden Layers: {self.num_hidden_layers.get()}")
        print(f"Layer Width: {self.layer_width.get()}")


if __name__ == '__main__':
    ui = NNConfigUI()  # Instantiate your Tkinter class
    ui.mainloop()      # Start the main loop

