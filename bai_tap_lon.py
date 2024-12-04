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
        self.mode_combo = ttk.Combobox(
            root, values=["Morphological Operations", "Image Filters"])
        self.mode_combo.set("Select Mode")
        self.mode_combo.pack(pady=10)
        self.mode_combo.bind("<<ComboboxSelected>>", self.switch_mode)

        # Load button
        self.load_button = Button(
            root, text="Load Image", command=self.load_image)
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

        morph_widgets['threshold_slider'] = Scale(
            self.root, from_=0, to=255, orient=tk.HORIZONTAL, label="Binary Threshold")
        morph_widgets['threshold_slider'].set(127)

        morph_widgets['operation_combo'] = ttk.Combobox(
            self.root, values=["Erosion", "Dilation", "Opening", "Closing"])
        morph_widgets['operation_combo'].current(0)

        morph_widgets['shape_combo'] = ttk.Combobox(
            self.root, values=["Rectangle", "Ellipse", "Cross"])
        morph_widgets['shape_combo'].current(0)

        morph_widgets['kernel_size_scale'] = Scale(
            self.root, from_=1, to=21, orient=tk.HORIZONTAL, label="Kernel Size", resolution=2)
        morph_widgets['kernel_size_scale'].set(3)

        morph_widgets['apply_button'] = Button(
            self.root, text="Apply Morphology", command=self.apply_morphology)

        for widget in morph_widgets.values():
            widget.pack_forget()

        return morph_widgets

    def update_filter_widgets(self, event=None):
        filter_type = self.filter_widgets['filter_combo'].get()
        if filter_type == "Median Filter":
            self.filter_widgets['kernel_shape_combo'].pack_forget()
        else:
            self.filter_widgets['kernel_shape_combo'].pack()

    def create_filter_widgets(self):
        filter_widgets = {}
    
        filter_widgets['filter_combo'] = ttk.Combobox(
            self.root, values=["Median Filter", "MIN Filter", "MAX Filter"])
        filter_widgets['filter_combo'].set("Median Filter")
        filter_widgets['filter_combo'].bind("<<ComboboxSelected>>", self.update_filter_widgets)
    
        filter_widgets['kernel_shape_combo'] = ttk.Combobox(
            self.root, values=["Rectangle", "Ellipse", "Cross"])
        filter_widgets['kernel_shape_combo'].set("Rectangle")
    
        filter_widgets['kernel_size_scale'] = Scale(
            self.root, from_=1, to=21, orient=tk.HORIZONTAL, label="Kernel Size", resolution=2)
        filter_widgets['kernel_size_scale'].set(3)
    
        filter_widgets['apply_button'] = Button(
            self.root, text="Apply Filter", command=self.apply_filter)
    
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
            self.update_filter_widgets()  # Cập nhật UI ngay khi chuyển chế độ

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

            # Check if the image is grayscale by ensuring it has only one channel or all channels are equal
            if len(self.img.shape) == 2:  # Only one channel (grayscale)
                self.is_gray = True
            elif len(self.img.shape) == 3 and self.img.shape[2] == 3:
                # Check if all channels are the same, indicating grayscale
                self.is_gray = np.all(self.img[:, :, 0] == self.img[:, :, 1]) and np.all(
                    self.img[:, :, 0] == self.img[:, :, 2])
            else:
                self.is_gray = False

            self.display_image(self.img)

    def display_image(self, img):
        if img is None:
            print("Image not loaded properly.")
            return

        # Determine the color conversion based on grayscale check
        if self.is_gray:
            if len(img.shape) == 2:  # Directly grayscale
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:  # Check for images that may appear grayscale but have 3 channels
                img_rgb = cv2.cvtColor(cv2.cvtColor(
                    img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        im_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.image_label.configure(image=imgtk)
        self.image_label.image = imgtk

    def apply_morphology(self):
        if self.image_path is None:
            return

        # Convert to grayscale for morphological operations
        gray_image = cv2.cvtColor(
            self.img, cv2.COLOR_BGR2GRAY) if not self.is_gray else self.img
        _, binary_image = cv2.threshold(
            gray_image, self.morph_widgets['threshold_slider'].get(), 255, cv2.THRESH_BINARY)

        shape = self.morph_widgets['shape_combo'].get()
        self.kernel_shape = {'Rectangle': cv2.MORPH_RECT, 'Ellipse': cv2.MORPH_ELLIPSE,
                             'Cross': cv2.MORPH_CROSS}.get(shape, cv2.MORPH_RECT)

        # Ensure kernel size is odd
        kernel_size = self.morph_widgets['kernel_size_scale'].get()
        if kernel_size % 2 == 0:
            kernel_size += 1  # Adjust to the next odd number

        kernel = cv2.getStructuringElement(
            self.kernel_shape, (kernel_size, kernel_size))
        print(kernel)

        operation = self.morph_widgets['operation_combo'].get()
        if operation == 'Erosion':
            processed_image = cv2.erode(binary_image, kernel)
        elif operation == 'Dilation':
            processed_image = cv2.dilate(binary_image, kernel)
        elif operation == 'Opening':
            processed_image = cv2.morphologyEx(
                binary_image, cv2.MORPH_OPEN, kernel)
        elif operation == 'Closing':
            processed_image = cv2.morphologyEx(
                binary_image, cv2.MORPH_CLOSE, kernel)

        self.display_image(processed_image)

    def apply_filter(self):
        if self.image_path is None:
            return

        kernel_size = self.filter_widgets['kernel_size_scale'].get()
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd

        filter_type = self.filter_widgets['filter_combo'].get()

        # Hide kernel shape combobox for Median Filter
        if filter_type == "Median Filter":
            self.filter_widgets['kernel_shape_combo'].pack_forget()
        else:
            self.filter_widgets['kernel_shape_combo'].pack()

        kernel_shape_name = self.filter_widgets['kernel_shape_combo'].get(
        ) if filter_type != "Median Filter" else "Rectangle"
        kernel_shape = {'Rectangle': cv2.MORPH_RECT, 'Ellipse': cv2.MORPH_ELLIPSE,
                        'Cross': cv2.MORPH_CROSS}.get(kernel_shape_name, cv2.MORPH_RECT)

        # Create the kernel
        kernel = cv2.getStructuringElement(
            kernel_shape, (kernel_size, kernel_size))

        # Apply filter based on selection
        if filter_type == "Median Filter":
            if self.is_gray:
                filtered_image = cv2.medianBlur(self.img, kernel_size)
            else:
                # Apply median filter to each color channel separately
                #cũng tác RGB tương tự
                channels = cv2.split(self.img)
                filtered_channels = [cv2.medianBlur(
                    ch, kernel_size) for ch in channels]
                filtered_image = cv2.merge(filtered_channels)

        elif filter_type == "MIN Filter":
            if self.is_gray:
                # borderType=cv2.BORDER_CONSTANT, borderValue=255
                filtered_image = cv2.erode(self.img, kernel)
            else:
                # Apply MIN filter to each color channel separately
                # Hàm cv2.split() tách ảnh self.img thành ba kênh màu riêng biệt: Red (R), Green (G), và Blue (B). Mỗi kênh này là một ma trận 2D.
                # print(channels[0])  # Kênh Blue
                # print(channels[1])  # Kênh Green
                # print(channels[2])  # Kênh Red
                channels = cv2.split(self.img)
                min_filtered_channels = [
                    cv2.erode(ch, kernel) for ch in channels]
                #hợp nhất 3 ma trận RGB lại
                filtered_image = cv2.merge(min_filtered_channels)

        elif filter_type == "MAX Filter":
            if self.is_gray:
                filtered_image = cv2.dilate(self.img, kernel)
            else:
                # Apply MAX filter to each color channel separately
                channels = cv2.split(self.img)
                max_filtered_channels = [
                    cv2.dilate(ch, kernel) for ch in channels]
                filtered_image = cv2.merge(max_filtered_channels)

        self.display_image(filtered_image)


# Create the main window
root = tk.Tk()
app = CombinedApp(root)
root.mainloop()
