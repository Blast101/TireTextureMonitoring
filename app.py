import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load your trained model without compiling (for inference only)
model = tf.keras.models.load_model('model3.hdf5', compile=False)

# Define the classes
classes = {0: 'Normal Tire', 1: 'Cracked Tire'}

# Set the threshold for classification
THRESHOLD = 0.5

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((300, 300), Image.LANCZOS)  # Use LANCZOS for high-quality resizing
    img = np.array(img).astype('float32') / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

# Function to make a prediction
def predict_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    class_index = 1 if prediction[0][0] >= THRESHOLD else 0
    return classes[class_index]

# Function to handle file selection and prediction
def choose_file():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
    )
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        prediction_result.set(predict_image(file_path))

# Create main window
root = tk.Tk()
root.title("Tire Texture Classifier")
root.geometry("800x600")

# Background image
background_image = Image.open("background_image.jpg")
background_image = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Title label
title_label = tk.Label(root, text="Tire Texture Classifier",
                       font=("Helvetica", 24, "bold"), bg="#b3e5fc")
title_label.pack(pady=10)

# Upload button with icon
upload_icon = Image.open("upload.png").resize((30, 30), Image.LANCZOS)
upload_icon = ImageTk.PhotoImage(upload_icon)
upload_button = ttk.Button(root, text="Upload Image",
                           image=upload_icon, compound="left", command=choose_file)
upload_button.pack(pady=20)

# Image display panel
panel = tk.Label(root)
panel.pack()

# Prediction result display
prediction_result = tk.StringVar()
prediction_label = tk.Label(root, textvariable=prediction_result,
                            font=("Helvetica", 18), fg="green")
prediction_label.pack(pady=20)

# Run the GUI
if __name__ == "__main__":
    root.mainloop()
