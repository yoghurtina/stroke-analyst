import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import ImageTk, Image
import image_segmentation as seg
import os

class App:
    def __init__(self, master):
        self.master = master
        master.title("Image Segmentation")

        # Create the buttons
        self.import_button = tk.Button(master, text="Import Image", command=self.import_image)
        self.segment_button = tk.Button(master, text="Segment Image", command=self.segment_image)
        # self.download_button = tk.Button(master, text="Download Image", command=self.download_image)

        # Create the image canvas
        self.image_canvas = tk.Canvas(master, width=512, height=512)
        self.segmented_canvas = tk.Canvas(master, width=512, height=512)

        # Create the layout
        self.import_button.pack()
        self.segment_button.pack()
        # self.download_button.pack()
        self.image_canvas.pack(side=tk.LEFT, padx=(20, 10))
        self.segmented_canvas.pack(side=tk.LEFT, padx=(10, 20))

        # Initialize the image and result variables
        self.image = None
        self.result = None

   
    def import_image(self):
        # Prompt the user to select an image file
        file_path = filedialog.askopenfilename()
        if file_path:
            # Load the image using OpenCV and convert it to a Tkinter PhotoImage
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            self.image_tk = ImageTk.PhotoImage(image)
            # Update the image canvas
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            self.image = cv2.imread(file_path)



    def segment_image(self):
        if self.image is not None:
            # Convert the image to a PIL Image object
            image = Image.fromarray(self.image)

            # Save the image to a temporary file
            temp_file = ".temp.jpg"
            image.save(temp_file)

            # Perform the segmentation on the loaded image
            result = seg.segmentation(temp_file)
            
            # Convert the result to a Tkinter PhotoImage
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            result = Image.fromarray(result)
            result = ImageTk.PhotoImage(result)
            # Update the image canvas
            self.segmented_canvas.delete("all")
            self.segmented_canvas.create_image(0, 0, anchor=tk.NW, image=result)
            self.result = result
            
            # Remove the temporary file
            os.remove(temp_file)



    # def download_image(self):
    #     if self.result is not None:
    #         # Prompt the user to select a filename and save the processed image
    #         file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
    #         if file_path:
    #             result = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
    #             cv2.imwrite(file_path, result)


root = tk.Tk()
app = App(root)
root.mainloop()
