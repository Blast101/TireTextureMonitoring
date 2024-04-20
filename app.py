import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('model3.hdf5')

# Define the classes
classes = {0: 'Normal Tire', 1: 'Cracked Tire'}

# Set the threshold for classification
THRESHOLD = 0.5

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((300, 300), Image.ANTIALIAS)  # Resize to the model's expected input size
    img = np.array(img) / 255.0  # Normalize pixel values to between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make a prediction
def predict_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    class_index = 1 if prediction[0][0] >= THRESHOLD else 0
    predicted_class = classes[class_index]
    return predicted_class

# Function to handle the file selection and prediction
def choose_file():
    file_path = filedialog.askopenfilename(title="Select an image",
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        # Display the selected image
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        # Make a prediction
        prediction_result.set(predict_image(file_path))

# Create the main window
root = tk.Tk()
root.title("Tire Texture Classifier")

# Set the window size
root.geometry("800x600")

# Set the background image
background_image = Image.open("background_image.jpg")  # Replace with the actual path to your background image
background_image = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Create a title label
title_label = tk.Label(root, text="Tire Texture Classifier", font=("Helvetica", 24, "bold"), bg="#b3e5fc")
title_label.pack(pady=10)

# Create a dropdown for image upload
upload_icon = Image.open("upload.png")  # Replace with the actual path to your upload icon
upload_icon = ImageTk.PhotoImage(upload_icon.resize((30, 30), Image.ANTIALIAS))
upload_button = ttk.Button(root, text="Upload Image", image=upload_icon, compound="left", command=choose_file)
upload_button.pack(pady=20)

# Display the selected image
panel = tk.Label(root)
panel.pack()

# Display the prediction result
prediction_result = tk.StringVar()
prediction_label = tk.Label(root, textvariable=prediction_result, font=("Helvetica", 18), fg="green")
prediction_label.pack(pady=20)

# Run the GUI
root.mainloop()
