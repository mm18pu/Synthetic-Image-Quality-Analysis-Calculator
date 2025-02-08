import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io, color, transform
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
#import cv2
import csv

class ImageSNRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Image Quality Analysis Calculator")

        self.image_path = None
        self.ground_truth_path = None

        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        self.ax[0].set_title('Image')
        self.ax[1].set_title('Ground Truth')

        self.roi_rect_image = None
        self.roi_rect_ground_truth = None
        self.roi_coords = [None, None]  # Initialize with placeholders for image and ground truth ROIs

        self.image = None
        self.image_display = None
        self.ground_truth = None

        # Create a frame for the buttons (horizontal layout)
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Add buttons in the horizontal frame
        self.load_button = tk.Button(button_frame, text="Load Images", command=self.load_images)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.toggle_button = tk.Button(button_frame, text="Toggle Color Mode", command=self.toggle_color_mode)
        self.toggle_button.pack(side=tk.LEFT, padx=5)
        self.color_mode = "RGB"

        self.calc_snr_button = tk.Button(button_frame, text="Calculate SNR", command=self.compute_snr_and_display)
        self.calc_snr_button.pack(side=tk.LEFT, padx=5)

        self.calc_ssim_button = tk.Button(button_frame, text="Calculate SSIM", command=self.compute_ssim_and_display)
        self.calc_ssim_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(button_frame, text="Save Image & ROI", command=self.save_image_pair_and_roi_to_csv)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_application)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.quit_button = tk.Button(button_frame, text="Quit", command=self.quit)
        self.quit_button.pack(side=tk.LEFT, padx=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.drawing_box = False  # Flag to indicate if ROI drawing is active
        self.drawing_ground_truth_box = False  # Flag for ground truth ROI
        # Add a button to call this method:

    def save_image_pair_and_roi_to_csv(self):
        if self.image_path and self.ground_truth_path and self.roi_coords[0] and self.roi_coords[1]:
            with open('image_roi_data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.image_path, self.ground_truth_path, *self.roi_coords[0], *self.roi_coords[1]])
            messagebox.showinfo("Saved", "Image pair and ROI data saved to CSV.")
        else:
            messagebox.showerror("Error", "Please load images and select ROIs before saving.")


    def load_images(self):
        self.image_path = filedialog.askopenfilename(title="Select Image")
        self.ground_truth_path = filedialog.askopenfilename(title="Select Ground Truth Image")

        if self.image_path and self.ground_truth_path:
            self.image = io.imread(self.image_path)
            self.ground_truth = io.imread(self.ground_truth_path)
            
            # Remove alpha channel if present
            if self.image.shape[2] == 4:
                self.image = color.rgba2rgb(self.image)
            if self.ground_truth.shape[2] == 4:
                self.ground_truth = color.rgba2rgb(self.ground_truth)
            
            self.image_display = self.image  # Initial display is RGB

            self.update_displayed_images()

            self.canvas.draw()
            messagebox.showinfo("Instructions", "Click once to start drawing a region of interest (ROI) on the Image. Click again to set its size. Then, repeat the same process for the Ground Truth image.")

    def update_displayed_images(self):
        if self.color_mode == "RGB":
            self.ax[0].imshow(self.image_display)
            self.ax[1].imshow(self.ground_truth)
        elif self.color_mode == "Grayscale":
            self.ax[0].imshow(color.rgb2gray(self.image_display), cmap='gray')
            self.ax[1].imshow(color.rgb2gray(self.ground_truth), cmap='gray')

        if self.roi_rect_image:
            self.ax[0].add_patch(self.roi_rect_image)
        if self.roi_rect_ground_truth:
            self.ax[1].add_patch(self.roi_rect_ground_truth)

        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes:
            if event.inaxes == self.ax[0]:  # Clicked on the first image
                if not self.drawing_box:
                    self.roi_coords[0] = (event.xdata, event.ydata)
                    self.drawing_box = True
                else:
                    x0, y0 = self.roi_coords[0][:2]
                    x1, y1 = event.xdata, event.ydata
                    self.roi_coords[0] = (x0, y0, abs(x1 - x0), abs(y1 - y0))
                    self.draw_roi_on_image(x0, y0, abs(x1 - x0), abs(y1 - y0))
                    self.drawing_box = False
            elif event.inaxes == self.ax[1]:  # Clicked on the ground truth image
                if not self.drawing_ground_truth_box:
                    self.roi_coords[1] = (event.xdata, event.ydata)
                    self.drawing_ground_truth_box = True
                else:
                    x0, y0 = self.roi_coords[1][:2]
                    x1, y1 = event.xdata, event.ydata
                    self.roi_coords[1] = (x0, y0, abs(x1 - x0), abs(y1 - y0))
                    self.draw_roi_on_ground_truth(x0, y0, abs(x1 - x0), abs(y1 - y0))
                    self.drawing_ground_truth_box = False

    def on_release(self, event):
        pass

    def draw_roi_on_image(self, x, y, width, height):
        if self.roi_rect_image:
            self.roi_rect_image.remove()

        self.roi_rect_image = Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        self.ax[0].add_patch(self.roi_rect_image)
        self.canvas.draw()

    def draw_roi_on_ground_truth(self, x, y, width, height):
        if self.roi_rect_ground_truth:
            self.roi_rect_ground_truth.remove()

        self.roi_rect_ground_truth = Rectangle((x, y), width, height, linewidth=2, edgecolor='g', facecolor='none')
        self.ax[1].add_patch(self.roi_rect_ground_truth)
        self.canvas.draw()

    def compute_snr_and_display(self):
        if self.roi_coords[0] is not None and self.roi_coords[1] is not None:
            x0_img, y0_img, width_img, height_img = self.roi_coords[0]
            x0_gt, y0_gt, width_gt, height_gt = self.roi_coords[1]

            # Convert to integers
            x_img, y_img = int(x0_img), int(y0_img)
            width_img, height_img = int(width_img), int(height_img)
            x_gt, y_gt = int(x0_gt), int(y0_gt)
            width_gt, height_gt = int(width_gt), int(height_gt)

            # Ensure correct order of coordinates for image ROI
            x_img, y_img = int(x0_img), int(y0_img)
            if width_img < 0:
                x_img += width_img
                width_img = abs(width_img)
            if height_img < 0:
                y_img += height_img
                height_img = abs(height_img)

            # Ensure correct order of coordinates for ground truth ROI
            x_gt, y_gt = int(x0_gt), int(y0_gt)
            if width_gt < 0:
                x_gt += width_gt
                width_gt = abs(width_gt)
            if height_gt < 0:
                y_gt += height_gt
                height_gt = abs(height_gt)

            # ROI for image
            roi_image = self.image[y_img:y_img + height_img, x_img:x_img + width_img]

            # ROI for ground truth
            roi_ground_truth = self.ground_truth[y_gt:y_gt + height_gt, x_gt:x_gt + width_gt]

            # Resize to match sizes if necessary
            if roi_image.shape[:2] != roi_ground_truth.shape[:2]:
                if roi_image.shape[0] * roi_image.shape[1] > roi_ground_truth.shape[0] * roi_ground_truth.shape[1]:
                    roi_image = transform.resize(roi_image, roi_ground_truth.shape[:2])
                else:
                    roi_ground_truth = transform.resize(roi_ground_truth, roi_image.shape[:2])

            # Convert to grayscale if needed
            if self.color_mode == "Grayscale":
                roi_image = color.rgb2gray(roi_image)
                roi_ground_truth = color.rgb2gray(roi_ground_truth)

            # snr_db = psnr(roi_ground_truth, roi_image)
            #"""
            # Compute SNR
            signal = np.mean(roi_ground_truth)
            noise = np.std(roi_image - roi_ground_truth)

            if noise < np.finfo(float).eps:
                snr_db = np.inf
            else:
                snr = signal / noise
                snr_db = 10 * np.log10(snr)
                #print(10*np.log10(signal), 10*np.log10(noise))
            #"""
            messagebox.showinfo("SNR (dB)", f"SNR (dB) in selected ROI: {snr_db:.2f} dB")
        else:
            messagebox.showerror("Error", "Please select ROIs on both the Image and the Ground Truth before calculating SNR.")

    def compute_ssim_and_display(self):
        if self.roi_coords[0] is not None and self.roi_coords[1] is not None:
            x0_img, y0_img, width_img, height_img = self.roi_coords[0]
            x0_gt, y0_gt, width_gt, height_gt = self.roi_coords[1]

            # Convert to integers
            x_img, y_img = int(x0_img), int(y0_img)
            width_img, height_img = int(width_img), int(height_img)
            x_gt, y_gt = int(x0_gt), int(y0_gt)
            width_gt, height_gt = int(width_gt), int(height_gt)

            # Ensure correct order of coordinates for image ROI
            x_img, y_img = int(x0_img), int(y0_img)
            if width_img < 0:
                x_img += width_img
                width_img = abs(width_img)
            if height_img < 0:
                y_img += height_img
                height_img = abs(height_img)

            # Ensure correct order of coordinates for ground truth ROI
            x_gt, y_gt = int(x0_gt), int(y0_gt)
            if width_gt < 0:
                x_gt += width_gt
                width_gt = abs(width_gt)
            if height_gt < 0:
                y_gt += height_gt
                height_gt = abs(height_gt)

            # ROI for image
            roi_image = self.image[y_img:y_img + height_img, x_img:x_img + width_img]

            # ROI for ground truth
            roi_ground_truth = self.ground_truth[y_gt:y_gt + height_gt, x_gt:x_gt + width_gt]

            # Resize to match sizes if necessary
            if roi_image.shape[:2] != roi_ground_truth.shape[:2]:
                if roi_image.shape[0] * roi_image.shape[1] > roi_ground_truth.shape[0] * roi_ground_truth.shape[1]:
                    roi_image = transform.resize(roi_image, roi_ground_truth.shape[:2])
                else:
                    roi_ground_truth = transform.resize(roi_ground_truth, roi_image.shape[:2])

            # Convert to grayscale if needed
            if self.color_mode == "Grayscale":
                roi_image = color.rgb2gray(roi_image)
                roi_ground_truth = color.rgb2gray(roi_ground_truth)

            # Compute SSIM
            #ssim_index = ssim(roi_image, roi_ground_truth, data_range=roi_image.max() - roi_image.min(), multichannel=(self.color_mode == "RGB"))
            ssim_index = ssim(
                    roi_image, 
                    roi_ground_truth, 
                    data_range=roi_image.max() - roi_image.min(), 
                    channel_axis=-1 if self.color_mode == "RGB" else None
                )


            messagebox.showinfo("SSIM", f"SSIM Index in selected region: {ssim_index:.4f}", parent=self.root)

            # Display the ROI images
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(roi_image, cmap='gray' if self.color_mode == 'Grayscale' else None)
            ax[0].set_title('ROI Image')
            ax[1].imshow(roi_ground_truth, cmap='gray' if self.color_mode == 'Grayscale' else None)
            ax[1].set_title('ROI Ground Truth')
            plt.show()
        else:
            messagebox.showerror("Error", "Please draw ROIs on both images first.", parent=self.root)


    def toggle_color_mode(self):
        if self.color_mode == "RGB":
            self.color_mode = "Grayscale"
        else:
            self.color_mode = "RGB"
        self.update_displayed_images()

    def reset_application(self):
        self.image_path = None
        self.ground_truth_path = None
        self.roi_coords = [None, None]
        self.roi_rect_image = None
        self.roi_rect_ground_truth = None
        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[0].set_title('Image')
        self.ax[1].set_title('Ground Truth')
        self.canvas.draw()

    def quit(self):
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSNRApp(root)
    root.mainloop()