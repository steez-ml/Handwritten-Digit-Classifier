import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import keras

class DigitClassifierGUI:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()

        # create a drawing canvas
        self.canvas = tk.Canvas(self.window, width=600, height=600, bg='white')
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<B1-Motion>", self.draw_digit)

        # Initialize the image
        self.image = Image.new('RGB', (600, 600), 'black')
        self.draw = ImageDraw.Draw(self.image)

        # create a label and a button
        self.label = tk.Label(self.window, text="Draw a digit and press predict")
        self.pred_button = tk.Button(self.window, text="Predict", command=self.predict, height=5, width=10)
        self.clear_button = tk.Button(self.window, text="Clear", command=self.clear, height=5, width=10)

        # add the label and button to the window
        self.label.grid(row=1, column=0)
        self.pred_button.grid(row=1, column=1)
        self.clear_button.grid(row=0, column=1)

    def draw_digit(self, event):
        size = 21  # size of one 'pixel'
        x = event.x // size * size
        y = event.y // size * size
        self.canvas.create_rectangle(x, y, x + size, y + size, fill='black')
        self.draw.rectangle([(x, y), (x + size, y + size)], fill='white')

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new('RGB', (600, 600), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # resize the image and convert it to grayscale
        image_resized = self.image.resize((28, 28)).convert('L')
        image_array = np.array(image_resized)

        # reshape the array to match the model's input shape
        image_array = image_array.reshape(1, 28, 28, 1)

        # display the image
        fig = Figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(image_array.reshape(28, 28), cmap='gray')
        canvas = FigureCanvasTkAgg(fig, master=self.window)  
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=2)

        # make a prediction
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)

        # update the label
        self.label.config(text=f"Predicted digit: {predicted_digit}, confidence: {confidence:.2f}")
        

    def run(self):
        self.window.mainloop()

# load the trained model
model = keras.models.load_model('trained models/mlp_augmented.md5')

gui = DigitClassifierGUI(model)
gui.run()
