import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageTk
import keras
import matplotlib.pyplot as plt

# Assuming 'decoder.h5' is the correct path to the model
model = keras.models.load_model('decoder.h5')

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

    # Draw the grid of points
    grid_spacing = 500 // 500
    for i in range(500):
        for j in range(500):
            if i % grid_spacing == 0 and j % grid_spacing == 0:
                # canvas.create_rectangle(i // grid_spacing, j // grid_spacing, (i // grid_spacing) + 1, (j // grid_spacing) + 1, fill="gray")
                canvas.create_text(i // grid_spacing, j // grid_spacing, text=".", font=("Helvetica", 1), fill="white")

    # Create a label for the predicted image
    default_image = Image.new("RGB", (300, 300), color="white")  # Change the default image size to 300x300
    default_photo = ImageTk.PhotoImage(default_image)
    image_label = tk.Label(frame, image=default_photo)
    image_label.image = default_photo
    image_label.pack(side=tk.LEFT)

    # Bind the motion (hover) event to the canvas
    canvas.bind("<Motion>", on_hover)

    # Start the main event loop
    root.mainloop()
