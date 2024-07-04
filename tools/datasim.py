import tkinter as tk
from tkinter import ttk

class Rectangle:
    def __init__(self, canvas, x1, y1, x2, y2, width, height, outline="blue", fill="black", thickness=5):
        self.canvas = canvas
        self.rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline, fill=fill, width=thickness)
        self.thickness = thickness
        self.x1, self.y1 = x1, y1
        self.width = width
        self.height = height

    def update_width(self, width):
        self.width = width
        self.canvas.coords(self.rect, self.x1, self.y1, self.x1 + self.width, self.y1 + self.height)

    def update_height(self, height):
        self.height = height
        self.canvas.coords(self.rect, self.x1, self.y1, self.x1 + self.width, self.y1 + self.height)

    def update_outline_width(self, thickness):
        self.thickness = thickness
        self.canvas.itemconfig(self.rect, width=thickness)


class SliderGroup:
    def __init__(self, master, rectangle, row, defaultwidth, defaultheight):
        self.rectangle = rectangle

        # Create sliders
        self.width_slider = ttk.Scale(master, from_=1, to=1920, orient=tk.HORIZONTAL, command=self.update_width)
        self.width_slider.set(defaultwidth)
        self.width_slider.grid(row=row, column=0, padx=10, pady=5)

        self.height_slider = ttk.Scale(master, from_=1, to=1080, orient=tk.HORIZONTAL, command=self.update_height)
        self.height_slider.set(defaultheight)
        self.height_slider.grid(row=row, column=1, padx=10, pady=5)

        self.pos_slider = ttk.Scale(master, from_=1, to=20, orient=tk.HORIZONTAL, command=self.update_outline_width)
        self.pos_slider.set(2)
        self.pos_slider.grid(row=row, column=2, padx=10, pady=5)

    def update_width(self, val):
        size = int((float(val)*2)/3)
        self.rectangle.update_width(size)

    def update_height(self, val):
        size = int((float(val)*2)/3)
        self.rectangle.update_height(size)

    def update_outline_width(self, val):
        width = int(float(val))
        self.rectangle.update_outline_width(width)


class RectangleAdjusterApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Data Simulation Tool v1.0")

        # Create a canvas
        self.canvas = tk.Canvas(self, width=1280, height=720, bg="black")
        self.canvas.pack()

        # Frame for sliders
        self.slider_frame = ttk.Frame(self)
        self.slider_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create rectangles and their paired sliders
        self.rectangles = []
        self.sliders = []

        self.create_rectangle_with_sliders(50, 50, 150, 150, 0, "red", 100, 150)
        self.create_rectangle_with_sliders(200, 200, 300, 300, 1, "purple", 150, 50)

    def create_rectangle_with_sliders(self, x1, y1, x2, y2, row, outline, width, height):
        rect = Rectangle(self.canvas, x1, y1, x2, y2, outline=outline, width=width, height=height)
        self.rectangles.append(rect)
        sliders = SliderGroup(self.slider_frame, rect, row, defaultwidth=width, defaultheight=height)
        self.sliders.append(sliders)


if __name__ == "__main__":
    app = RectangleAdjusterApp()
    app.mainloop()
