import tkinter as tk
from PIL import Image, ImageGrab
import numpy as np
import tensorflow as tf

# loading a trained model
model = tf.keras.models.load_model('digits_recognition_cnn.keras')

# a Tkinter window
window = tk.Tk()
window.title("Handwritten Digit Recognition")


# function to get and process the image
def process_image(filename):
    image = Image.open(filename).convert('L')
    width, height = image.size
    image = image.crop((6, 6, width - 6, height - 6))
    image = image.resize((28, 28))
    image.save(filename)
    return image


def getter(widget):
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()
    ImageGrab.grab().crop((x, y, x1, y1)).save("image.png")
    image = process_image('image.png')
    return np.array(image).reshape(28, 28)


# canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg="black",
                   borderwidth=2, relief="groove", highlightthickness=1, highlightbackground="black")
canvas.pack(side=tk.LEFT, padx=10, pady=10)

# an area to display the AI result
result_label = tk.Label(window, text="AI Result: ")
result_label.pack(side=tk.RIGHT, padx=10, pady=10)


# function to handle drawing on the canvas
def draw(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="white", width=15, outline="white")


def clean(event):
    canvas.delete('all')


canvas.bind("<B1-Motion>", draw)
canvas.bind("<Button-3>", clean)


# function to recognize the handwritten digit using AI model
def recognize_digit():
    input_image = getter(canvas) / 255
    prediction = model.predict(np.array([input_image]))[0]
    res = np.argmax(prediction)

    result_label.config(text=f"AI Result: {res}")


recognize_button = tk.Button(window, text="Recognize Digit", command=recognize_digit)
recognize_button.pack(side=tk.BOTTOM, pady=10)

window.mainloop()
