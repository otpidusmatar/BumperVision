import tkinter as tk
from tkinter import ttk
from math import sqrt
import numpy as np

class Rectangle:
    def __init__(self, canvas, x1, y1, x2, y2, width, height, outline="blue", fill="black", thickness=5):
        self.canvas = canvas
        self.rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline, fill=fill, width=thickness)
        self.thickness = thickness
        self.x1, self.y1 = x1, y1
        self.width = width
        self.height = height
        
    def update_box_status(self):
        self.canvas.coords(self.rect, self.x1, self.y1, self.x1 + self.width, self.y1 + self.height)

    def update_width(self, width):
        self.width = width
        self.update_box_status()

    def update_height(self, height):
        self.height = height
        self.update_box_status()

    def update_hpos(self, pos):
        self.x1 = pos
        self.update_box_status()

    def update_vpos(self, pos):
        self.y1 = pos
        self.update_box_status()

    def update_outline_width(self, thickness):
        self.thickness = thickness
        self.canvas.itemconfig(self.rect, width=thickness)

class VectorLines:
    def __init__(self, canvas, x1, y1, x2, y2, slope, distance, width, height):
        self.canvas = canvas
        self.x1, self.y1 = x1, y1
        self.slope = slope
        self.distance = distance
        self.width = width
        self.height = height

        self.pt1a = (x1, y1)
        self.pt2a = (x2, y1)
        self.pt3a = (x1, y2)
        self.pt4a = (x2, y2)

        self.pt1b = self.find_expected_new_pt(distance, slope, self.pt1a[0], self.pt1a[1])
        self.pt2b = self.find_expected_new_pt(distance, slope, self.pt2a[0], self.pt2a[1])
        self.pt3b = self.find_expected_new_pt(distance, slope, self.pt3a[0], self.pt3a[1])
        self.pt4b = self.find_expected_new_pt(distance, slope, self.pt4a[0], self.pt4a[1])

        self.line1 = self.canvas.create_line(self.pt1a[0], self.pt1a[1], self.pt1b[0], self.pt1b[1], fill="green", width=3)
        self.line2 = self.canvas.create_line(self.pt2a[0], self.pt2a[1], self.pt2b[0], self.pt2b[1], fill="green", width=3)
        self.line3 = self.canvas.create_line(self.pt3a[0], self.pt3a[1], self.pt3b[0], self.pt3b[1], fill="green", width=3)
        self.line4 = self.canvas.create_line(self.pt4a[0], self.pt4a[1], self.pt4b[0], self.pt4b[1], fill="green", width=3)

    def find_expected_new_pt(self, distance, slope, oldx, oldy):
        slope_squared = pow(slope, 2)
        y = sqrt((slope_squared*pow(distance, 2))/(slope_squared+1))+oldy
        x = ((y-oldy)/slope)+oldx
        try: return (int(x), int(y))
        except ValueError: return(int(oldx), int(oldy+distance))

    def update_points(self, x1, y1, width, height, slope, distance):
        self.pt1a = (x1, y1)
        self.pt2a = (x1 + width, y1)
        self.pt3a = (x1, y1 + height)
        self.pt4a = (x1 + width, y1 + height)

        self.pt1b = self.find_expected_new_pt(distance, slope, self.pt1a[0], self.pt1a[1])
        self.pt2b = self.find_expected_new_pt(distance, slope, self.pt2a[0], self.pt2a[1])
        self.pt3b = self.find_expected_new_pt(distance, slope, self.pt3a[0], self.pt3a[1])
        self.pt4b = self.find_expected_new_pt(distance, slope, self.pt4a[0], self.pt4a[1])

    def update_lines(self):
        self.update_points(self.x1, self.y1, self.width, self.height, self.slope, self.distance)
        self.canvas.coords(self.line1, self.pt1a[0], self.pt1a[1], self.pt1b[0], self.pt1b[1])
        self.canvas.coords(self.line2, self.pt2a[0], self.pt2a[1], self.pt2b[0], self.pt2b[1])
        self.canvas.coords(self.line3, self.pt3a[0], self.pt3a[1], self.pt3b[0], self.pt3b[1])
        self.canvas.coords(self.line4, self.pt4a[0], self.pt4a[1], self.pt4b[0], self.pt4b[1])

    def update_width(self, width):
        self.width = width
        self.update_lines()

    def update_height(self, height):
        self.height = height
        self.update_lines()

    def update_hpos(self, pos):
        self.x1 = pos
        self.update_lines()

    def update_vpos(self, pos):
        self.y1 = pos
        self.update_lines()

    def update_dir(self, slope):
        self.slope = slope
        self.update_lines()

    def update_dist(self, distance):
        self.distance = distance
        self.update_lines()

class SliderGroup:
    def __init__(self, master, rectangle, row, defaultx, defaulty, defaultwidth, defaultheight, vectorlines):
        self.rectangle = rectangle
        self.vectorlines = vectorlines

        # Create sliders
        self.width_slider = tk.Scale(master, from_=1, to=1920, orient=tk.HORIZONTAL, command=self.update_width)
        self.width_slider.set(defaultwidth)
        self.width_slider.grid(row=row, column=0, padx=10, pady=5)

        self.height_slider = tk.Scale(master, from_=1, to=1080, orient=tk.HORIZONTAL, command=self.update_height)
        self.height_slider.set(defaultheight)
        self.height_slider.grid(row=row, column=1, padx=10, pady=5)

        self.hshift_slider = tk.Scale(master, from_=1, to=1920, orient=tk.HORIZONTAL, command=self.update_hpos)
        self.hshift_slider.set(defaultx)
        self.hshift_slider.grid(row=row, column=2, padx=10, pady=5)

        self.vshift_slider = tk.Scale(master, from_=1, to=1080, orient=tk.HORIZONTAL, command=self.update_vpos)
        self.vshift_slider.set(defaulty)
        self.vshift_slider.grid(row=row, column=3, padx=10, pady=5)

        if self.vectorlines == None: 
            thick_slider_pos = 4
        else: 
            self.vectordir_slider = tk.Scale(master, from_=-5000, to=5000, orient=tk.HORIZONTAL, command=self.update_vectordir)
            self.vectordir_slider.set(defaulty)
            self.vectordir_slider.grid(row=row, column=4, padx=10, pady=5)

            self.vectorlen_slider = tk.Scale(master, from_=0, to=1000, orient=tk.HORIZONTAL, command=self.update_vectorlen)
            self.vectorlen_slider.set(defaulty)
            self.vectorlen_slider.grid(row=row, column=5, padx=10, pady=5)

            thick_slider_pos = 6

        self.thickness_slider = tk.Scale(master, from_=1, to=20, orient=tk.HORIZONTAL, command=self.update_outline_width)
        self.thickness_slider.set(5)
        self.thickness_slider.grid(row=row, column=thick_slider_pos, padx=10, pady=5)

    def update_width(self, val):
        size = int(round(((float(val)*2)/3), 0))
        self.rectangle.update_width(size)
        if self.vectorlines != None: 
            self.vectorlines.update_width(size)

    def update_height(self, val):
        size = int(round(((float(val)*2)/3), 0))
        self.rectangle.update_height(size)
        if self.vectorlines != None: 
            self.vectorlines.update_height(size)

    def update_hpos(self, val):
        pos = int(round(((float(val)*2)/3), 0))
        self.rectangle.update_hpos(pos)
        if self.vectorlines != None: 
            self.vectorlines.update_hpos(pos)

    def update_vpos(self, val):
        pos = int(round(((float(val)*2)/3), 0))
        self.rectangle.update_vpos(pos)
        if self.vectorlines != None: 
            self.vectorlines.update_vpos(pos)

    def update_vectordir(self, val):
        slope = int(round(float(val), 0))
        self.vectorlines.update_dir(slope)

    def update_vectorlen(self, val):
        distance = int(round(float(val), 0))
        self.vectorlines.update_dist(distance)

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
        self.vectorlines = []
        self.sliders = []

        self.create_rectangle_with_sliders(50, 50, 150, 150, 0, "red", 100, 150, -5, 20)
        self.create_rectangle_with_sliders(200, 200, 300, 300, 1, "purple", 150, 50, 0, 0)

    def create_rectangle_with_sliders(self, x1, y1, x2, y2, row, outline, width, height, slope, distance):
        rect = Rectangle(self.canvas, x1, y1, x2, y2, outline=outline, width=width, height=height)
        self.rectangles.append(rect)
        if slope == 0: 
            vect = None
        else:
            vect = VectorLines(self.canvas, x1, y1, x2, y2, slope, distance, width, height)
            self.vectorlines.append(vect)
        sliders = SliderGroup(self.slider_frame, rect, row, defaultx=x1, defaulty=y1, defaultwidth=width, defaultheight=height, vectorlines=vect)
        self.sliders.append(sliders)


if __name__ == "__main__":
    app = RectangleAdjusterApp()
    app.mainloop()
