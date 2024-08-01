""" Prerequisites: Python3, Numpy Library, Matplotlib Library """
""" To run the program please use the command "python3 perceptron.py" from the current directory """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

# Generate random x and y coordinates for points
np.random.seed(42)
x_points = np.random.randint(0, 1000, 20)
y_points = np.random.randint(0, 1000, 20)

# Define parameters for the linear decision boundary
a = 8 
b = 10

# Classify points based on their position relative to the line y = ax + b
classes = np.array([1 if y > (a * x + b) else -1 for x, y in zip(x_points, y_points)])

# Combine x and y coordinates into a single feature set
X = np.vstack((x_points, y_points)).T

# Implementation of Perceptron Model 
class Perceptron:
    '''
    Initializes the Perceptron Model with default parameters.

    Parameters:
    X (numpy array): Input features.
    y (numpy array): Target labels.
    learning_rate (float): Learning rate for updates, default is 0.0001.
    epochs (int): Number of training epochs, default is 1000.
    '''
    def __init__(self, X, y, learning_rate=0.0001, epochs=1000):
        self.X = X 
        self.y = y  
        self.w = np.random.rand(2)  # Initialize weights randomly
        self.b = np.random.rand()  # Initialize bias randomly
        self.learning_rate = learning_rate  
        self.epochs = epochs  
        self.fig, self.ax = plt.subplots(figsize=(10, 10))  # Setup the plot

    '''
    Step function used as the activation function.

    Parameters:
    z (float): Weighted sum of inputs.
    '''
    def activation_func(self, z):
        return 1 if z >= 0 else -1

    '''
    Updates weights and bias based on the error at a given index.

    Parameters:
    error (float): Error calculated as the difference between actual and predicted label.
    j (int): Index of the current data point in the feature array.
    '''
    def update_weights_bias(self, error, j):
        self.w += self.learning_rate * error * self.X[j]  # Update weights
        self.b += self.learning_rate * error  # Update bias

    '''
    Trains the model by adjusting weights and bias to minimize error.

    Parameters:
    None
    '''
    def train(self):
        errors = 0
        for j in range(len(self.X)):
            z = np.dot(self.X[j], self.w) + self.b  # Calculate the weighted sum
            y_pred = self.activation_func(z)  # Predict the class
            error = self.y[j] - y_pred  # Determine the error
            if error != 0:
                errors += 1  # Count misclassifications
                self.update_weights_bias(error, j)  # Update model
        return errors

    '''
    Visualizes the training process and decision boundary over epochs.

    Parameters:
    i (int): Current epoch number for updating the title.
    '''
    def plot_graph(self, i):
        self.ax.clear()
        self.ax.scatter(x_points[classes == 1], y_points[classes == 1], c='black', marker='o', label='Class 1')
        self.ax.scatter(x_points[classes == -1], y_points[classes == -1], c='none', edgecolors='black', marker='o', label='Class -1')
        x_line = np.linspace(0, 1000, 100)
        y_line = (-self.w[0] * x_line - self.b) / self.w[1]  # Calculate decision boundary
        self.ax.plot(x_line, y_line, 'g', label='Decision Boundary')
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(0, 1000)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.legend()
        errors = self.train()  # Train and get number of errors
        self.ax.set_title(f'Epoch {i+1}, Misclassifications: {errors}')
        print(f'Epoch {i + 1}: Misclassifications = {errors}')
        if errors == 0:
            print(f'Training complete after {i + 1} epochs.')
            self.anim.event_source.stop()

# Instantiate the Perceptron model and pass the dataset
model = Perceptron(X, classes)
# Set up the animation to visualize the training process
model.anim = animation.FuncAnimation(model.fig, model.plot_graph, frames=model.epochs, repeat=False)

# Display the plot
plt.show()
