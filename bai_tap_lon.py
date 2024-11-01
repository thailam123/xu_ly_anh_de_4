import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, Toplevel, Scale, Label, Button
from PIL import Image, ImageTk

class CombinedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Combined Image Processing Application")
        
        # Initialize variables
        self.img = None
        self.image_path = None
        self.kernel_shape = cv2.MORPH_RECT
        self.kernel_size = 3
        self.is_gray = False

        # Combobox for selection of operation mode
        self.mode_combo = ttk.Combobox(root, values=["Morphological Operations", "Image Filters"])
        self.mode_combo.set("Select Mode")
        self.mode_combo.pack(pady=10)
        self.mode_combo.bind("<<ComboboxSelected>>", self.switch_mode)

        # Load button
        self.load_button = Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Image display
        self.image_label = Label(root)
        self.image_label.pack()

        # Define morphology-specific widgets
        self.morph_widgets = self.create_morph_widgets()
        
        # Define filter-specific widgets
        self.filter_widgets = self.create_filter_widgets()

    def create_morph_widgets(self):
        morph_widgets = {}

        morph_widgets['threshold_slider'] = Scale(self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="Binary Threshold")
        morph_widgets['threshold_slider'].set(127)

        morph_widgets['operation_combo'] = ttk.Combobox(self.root, values=["Erosion", "Dilation", "Opening", "Closing"])
        morph_widgets['operation_combo'].current(0)
        
        morph_widgets['shape_combo'] = ttk.Combobox(self.root, values=["Rectangle", "Ellipse", "Cross"])
        morph_widgets['shape_combo'].current(0)
        
        morph_widgets['kernel_size_scale'] = Scale(self.root, from_=1, to=21, orient=tk.HORIZONTAL, label="Kernel Size")
        morph_widgets['kernel_size_scale'].set(3)

        morph_widgets['apply_button'] = Button(self.root, text="Apply Morphology", command=self.apply_morphology)
        
        for widget in morph_widgets.values():
            widget.pack_forget()

        return morph_widgets

    def create_filter_widgets(self):
        filter_widgets = {}
    
        filter_widgets['filter_combo'] = ttk.Combobox(self.root, values=["Median Filter", "MIN Filter", "MAX Filter"])
        filter_widgets['filter_combo'].set("Median Filter")
        
        # Set kernel size scale with odd numbers only (1, 3, 5, ..., 21)
        filter_widgets['kernel_size_scale'] = Scale(self.root, from_=1, to=21, orient=tk.HORIZONTAL, label="Kernel Size", resolution=2)
        filter_widgets['kernel_size_scale'].set(3)
        
        filter_widgets['apply_button'] = Button(self.root, text="Apply Filter", command=self.apply_filter)
        
        for widget in filter_widgets.values():
            widget.pack_forget()
    
        return filter_widgets


    def switch_mode(self, event):
        mode = self.mode_combo.get()
        if mode == "Morphological Operations":
            self.show_morph_widgets()
            self.hide_filter_widgets()
        elif mode == "Image Filters":
            self.show_filter_widgets()
            self.hide_morph_widgets()

    def show_morph_widgets(self):
        for widget in self.morph_widgets.values():
            widget.pack()

    def hide_morph_widgets(self):
        for widget in self.morph_widgets.values():
            widget.pack_forget()

    def show_filter_widgets(self):
        for widget in self.filter_widgets.values():
            widget.pack()

    def hide_filter_widgets(self):
        for widget in self.filter_widgets.values():
            widget.pack_forget()

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.img = cv2.imread(self.image_path)
            self.is_gray = len(self.img.shape) == 2 or self.img.shape[2] == 1
            self.display_image(self.img)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if self.is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.image_label.configure(image=imgtk)
        self.image_label.image = imgtk

    def apply_morphology(self):
        if self.image_path is None:
            return
        
        # Convert to grayscale for morphological operations
        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if not self.is_gray else self.img
        _, binary_image = cv2.threshold(gray_image, self.morph_widgets['threshold_slider'].get(), 255, cv2.THRESH_BINARY)
        
        shape = self.morph_widgets['shape_combo'].get()
        self.kernel_shape = {'Rectangle': cv2.MORPH_RECT, 'Ellipse': cv2.MORPH_ELLIPSE, 'Cross': cv2.MORPH_CROSS}.get(shape, cv2.MORPH_RECT)
        
        kernel_size = self.morph_widgets['kernel_size_scale'].get()
        kernel = cv2.getStructuringElement(self.kernel_shape, (kernel_size, kernel_size))
        
        operation = self.morph_widgets['operation_combo'].get()
        if operation == 'Erosion':
            processed_image = cv2.erode(binary_image, kernel)
        elif operation == 'Dilation':
            processed_image = cv2.dilate(binary_image, kernel)
        elif operation == 'Opening':
            processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        elif operation == 'Closing':
            processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        
        self.display_image(processed_image)

    def apply_filter(self):
        if self.image_path is None:
            return
        
        kernel_size = self.filter_widgets['kernel_size_scale'].get()
        if kernel_size % 2 == 0:
            kernel_size += 1

        filter_type = self.filter_widgets['filter_combo'].get()
        
        if filter_type == "Median Filter":
            if self.is_gray:
                filtered_image = cv2.medianBlur(self.img, kernel_size)
            else:
                # Apply median filter to each color channel separately
                channels = cv2.split(self.img)
                filtered_channels = [cv2.medianBlur(ch, kernel_size) for ch in channels]
                filtered_image = cv2.merge(filtered_channels)

        elif filter_type == "MIN Filter":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if self.is_gray:
                filtered_image = cv2.erode(self.img, kernel)
            else:
                # Apply MIN filter to each color channel separately
                channels = cv2.split(self.img)
                min_filtered_channels = [cv2.erode(ch, kernel) for ch in channels]
                filtered_image = cv2.merge(min_filtered_channels)

        elif filter_type == "MAX Filter":
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if self.is_gray:
                filtered_image = cv2.dilate(self.img, kernel)
            else:
                # Apply MAX filter to each color channel separately
                channels = cv2.split(self.img)
                max_filtered_channels = [cv2.dilate(ch, kernel) for ch in channels]
                filtered_image = cv2.merge(max_filtered_channels)
        
        self.display_image(filtered_image)

# Create the main window
root = tk.Tk()
app = CombinedApp(root)
root.mainloop()
