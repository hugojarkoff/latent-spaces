import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageTk
import keras
import matplotlib.pyplot as plt
from itertools import product

# Assuming 'decoder.h5' is the correct path to the model
model = keras.models.load_model('best_decoder.h5')
classifier_preds = np.load('classifier_preds.npy')

# Define a discrete color map with 10 distinct colors for each class
color_map = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'purple',
    4: 'orange',
    5: 'cyan',
    6: 'magenta',
    7: 'yellow',
    8: 'lime',
    9: 'brown',
}

# Create a mapping of class labels to text labels
class_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}

# Map the labels to their corresponding colors
colors = [color_map[label] for label in classifier_preds]



def predict_image(x, y):
    # Scale the coordinates to the range [0, 1]
    x_norm = x / 500
    y_norm = y / 500

    # Replace this with your DNN model prediction function
    # Assuming model.predict() returns a 2D array of shape (28, 28) for grayscale images
    predicted_image = model.predict([[x_norm, y_norm]])[0, :, :, 0]

    # Convert the predicted image from the range [0, 1] to the range [0, 255]
    predicted_image = (predicted_image * 255).astype(np.uint8)

    return predicted_image

def update_image(x, y):

    predicted_image = predict_image(x, y)

    # Create an ImageTk object from the grayscale image
    image = Image.fromarray(predicted_image)

    # Resize the image for better display while preserving the colormap
    image = image.resize((300, 300), resample=Image.NEAREST)  # Use NEAREST resampling to avoid blurring

    photo = ImageTk.PhotoImage(image)
    image_label.configure(image=photo)
    image_label.image = photo


def on_hover(event):
    x, y = event.x, event.y  # Use the raw hover coordinates
    update_image(x, y)

if __name__ == "__main__":
    # Create the main application window
    root = tk.Tk()
    root.title("Grid and Predicted Image")

    # Create a frame to hold both the canvas and the predicted image
    frame = tk.Frame(root)
    frame.pack()

    # Create the canvas for the grid (displaying 700x700 points on a 500x500 canvas)
    canvas = Canvas(frame, width=500, height=500, bg="white", scrollregion=(0, 0, 500, 500))
    canvas.pack(side=tk.LEFT)
    

    # Calculate the width and height of each rectangle
    rect_width = 1
    rect_height = 1

    for i, (x,y) in enumerate(product(np.arange(500), np.arange(500))) : 
        color = colors[i]
        canvas.create_rectangle(x * rect_width, y * rect_height, (x + 1) * rect_width, (y + 1) * rect_height, fill=color, outline='')

    # Create a label for the predicted image
    default_image = Image.new("RGB", (300, 300), color="white")  # Change the default image size to 300x300
    default_photo = ImageTk.PhotoImage(default_image)
    image_label = tk.Label(frame, image=default_photo)
    image_label.image = default_photo
    image_label.pack(side=tk.LEFT)

    # Create a label for the legend
    legend_frame = tk.Frame(frame)
    legend_frame.pack(side=tk.LEFT)

    legend_label = tk.Label(legend_frame, text='Legend:', font=('Helvetica', 14))
    legend_label.pack()

    # Create a canvas to display the color rectangles with their corresponding labels
    legend_canvas = tk.Canvas(legend_frame, width=150, height=300, bg="white")
    legend_canvas.pack()

    # Draw color rectangles with labels on the legend canvas
    rect_height = 30
    for i, label in class_labels.items():
        color = color_map[i]
        legend_canvas.create_rectangle(10, i * rect_height + 10, 40, (i + 1) * rect_height, fill=color)
        legend_canvas.create_text(60, (i + 0.5) * rect_height, text=label, anchor=tk.W)
        
    # Bind the motion (hover) event to the canvas
    canvas.bind("<Motion>", on_hover)

    # Start the main event loop
    root.mainloop()
