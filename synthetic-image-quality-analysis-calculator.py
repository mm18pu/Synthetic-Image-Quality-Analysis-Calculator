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
import pywt
import cv2
from phasepack.phasecong import phasecong  # FSIM dependency
import pywt.data
import sewar
from phasepack import phasecong



class ImageSNRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Image Quality Analysis Calculator")

        self.image_path = None
        self.ground_truth_path = None

        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        self.ax[0].set_title('Synthetic Image')
        self.ax[1].set_title('Reference Image')

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

        self.calc_snr_button = tk.Button(button_frame, text="Calculate PSNR", command=self.compute_snr_and_display)
        self.calc_snr_button.pack(side=tk.LEFT, padx=5)

        self.mbsnr_button = tk.Button(button_frame, text="Calculate MBSNR", command=self.compute_mbsnr)
        self.mbsnr_button.pack(side=tk.LEFT, padx=5)

        self.pbsnr_button = tk.Button(button_frame, text="Calculate PBSNR", command=self.compute_pbsnr)
        self.pbsnr_button.pack(side=tk.LEFT, padx=5)

        self.calc_ssim_button = tk.Button(button_frame, text="Calculate SSIM", command=self.compute_ssim_and_display)
        self.calc_ssim_button.pack(side=tk.LEFT, padx=5)

        self.cwssim_button = tk.Button(button_frame, text="Calculate CW-SSIM", command=self.compute_cwssim)
        self.cwssim_button.pack(side=tk.LEFT, padx=5)

        self.fsim_button = tk.Button(button_frame, text="Calculate FSIM", command=self.compute_fsim)
        self.fsim_button.pack(side=tk.LEFT, padx=5)

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
        self.image_path = filedialog.askopenfilename(title="Select Synthetic Image")
        self.ground_truth_path = filedialog.askopenfilename(title="Select Reference Image")

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
        self.ax[0].set_title('Synthetic Image')
        self.ax[1].set_title('Reference Image')
        self.canvas.draw()

    def quit(self):
        self.root.quit()
        self.root.destroy()

    def extract_rois(self):
        x0_img, y0_img, w_img, h_img = [int(v) for v in self.roi_coords[0]]
        x0_gt, y0_gt, w_gt, h_gt = [int(v) for v in self.roi_coords[1]]
        roi_image = self.image[y0_img:y0_img+h_img, x0_img:x0_img+w_img]
        roi_gt = self.ground_truth[y0_gt:y0_gt+h_gt, x0_gt:x0_gt+w_gt]
        if roi_image.shape[:2] != roi_gt.shape[:2]:
            roi_image = transform.resize(roi_image, roi_gt.shape[:2], preserve_range=True)
        return roi_image, roi_gt
    
    def calculate_mbsnr(roi_image, roi_ground_truth):
        signal = np.mean(roi_ground_truth)
        #print(len(roi_ground_truth))
        #print("here")
        #print(len(roi_image))
        noise = np.std(roi_image - roi_ground_truth)
        snr_value = 10 * np.log10(signal / noise) if noise > 0 else np.inf
        return snr_value, signal, noise


    def calculate_pbsnr(roi_image, roi_ground_truth):
        # Ensure inputs are NumPy arrays of type float for precision
        y = np.asarray(roi_ground_truth, dtype=np.float64)
        x = np.asarray(roi_image, dtype=np.float64)
        # Compute signal power (sum of squares of ground truth)
        signal_power = np.sum(y ** 2)
        signal = signal_power

        # Compute noise power (sum of squares of the difference)
        noise_power = np.sum(np.abs((y - x) ** 2))
        noise = noise_power
        # Avoid division by zero
        if noise_power == 0:
            return float('inf')  # Perfect match, infinite SNR

        # Calculate SNR in dB
        snr_value = 10 * np.log10(signal_power / noise_power)
        return snr_value, signal, noise


    def calculate_psnr(x, y, max_pixel_value=1):
        """ # Max pixel was 255
        Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images, normalizing if necessary.

        Args:
            y (numpy.ndarray): Ground truth image (signal).
            x (numpy.ndarray): Received image.
            max_pixel_value (float): Maximum possible pixel value of the image (e.g., 255 for 8-bit images).

        Returns:
            float: PSNR in decibels (dB).
        """
        # Ensure inputs are NumPy arrays of type float for precision
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)

        # Check and normalize images if necessary
        if y.max() > max_pixel_value or y.min() < 0:
            y = (y - y.min()) / (y.max() - y.min()) * max_pixel_value
        if x.max() > max_pixel_value or x.min() < 0:
            x = (x - x.min()) / (x.max() - x.min()) * max_pixel_value

        # Compute Mean Squared Error (MSE)
        mse = np.mean(np.abs((y - x) ** 2))

        # Avoid division by zero
        if mse == 0:
            return float('inf')  # Perfect match, infinite PSNR

        # Compute PSNR
        psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
        #print(psnr)
        return psnr, (max_pixel_value**2), mse
    
    def compute_ssim_and_display(self):
        roi_image, roi_gt = self.extract_rois()
        if self.color_mode == "Grayscale":
            roi_image = color.rgb2gray(roi_image)
            roi_gt = color.rgb2gray(roi_gt)
        ssim_index = ssim(roi_image, roi_gt, data_range=roi_image.max() - roi_image.min(), channel_axis=-1 if self.color_mode == "RGB" else None)
        messagebox.showinfo("SSIM", f"SSIM: {ssim_index:.4f}")

    def calculate_ssim_components(image, ground_truth, is_rgb=True):
        #print(image.max() - image.min())
        try:
            ssim_index, ssim_full = ssim(
                image, ground_truth, data_range=image.max() - image.min(),
                channel_axis=-1 if is_rgb else None, full=True
            )
        except:
            ssim_index, ssim_full = ssim(
                image, ground_truth, win_size=3, data_range=image.max() - image.min(),
                channel_axis=-1 if is_rgb else None, full=True
            )
        
        if is_rgb:
            luminance = ssim_full[0]  # Luminance for all channels
            contrast = ssim_full[1]   # Contrast for all channels
            structure = ssim_full[2]  # Structure for all channels

            luminance_r = luminance[..., 0].mean()
            contrast_r = contrast[..., 0].mean()
            structure_r = structure[..., 0].mean()

            luminance_g = luminance[..., 1].mean()
            contrast_g = contrast[..., 1].mean()
            structure_g = structure[..., 1].mean()

            luminance_b = luminance[..., 2].mean()
            contrast_b = contrast[..., 2].mean()
            structure_b = structure[..., 2].mean()

            return (
                ssim_index, 
                luminance_r, contrast_r, structure_r,  # Red channel
                luminance_g, contrast_g, structure_g,  # Green channel
                luminance_b, contrast_b, structure_b   # Blue channel
            )
        else:
            luminance = ssim_full[0]  # Luminance for grayscale
            contrast = ssim_full[1]   # Contrast for grayscale
            structure = ssim_full[2]  # Structure for grayscale

            return ssim_index, luminance.mean(), contrast.mean(), structure.mean()
        
    # CW-SSIM Calculation (Complex Wavelet SSIM)
    def complex_wavelet_transform(image):
        # Perform the complex wavelet transform using 'bior1.3' wavelet (you can choose other types)
        coeffs = pywt.dwt2(image, 'bior1.3')
        LL, (LH, HL, HH) = coeffs
        # Reconstruct the complex wavelet coefficients (Real and Imaginary parts)
        return LL, LH, HL, HH

    def calculate_cwssim(image, ground_truth):
        # Apply complex wavelet transforms
        LL1, LH1, HL1, HH1 = complex_wavelet_transform(image)
        LL2, LH2, HL2, HH2 = complex_wavelet_transform(ground_truth)
        
        # Compute SSIM for each wavelet coefficient with data_range specified
        win_size = min(7, min(LL1.shape))  # Choose 7 or the smallest dimension
        if win_size % 2 == 0:
            win_size = 3
        
        # LL (Low-Low) - Approximation coefficients: captures the low-frequency, large-scale structures.
        # LH (Low-High) - Horizontal detail coefficients: captures horizontal edges or high-frequency content.
        # HL (High-Low) - Vertical detail coefficients: captures vertical edges or high-frequency content.
        # HH (High-High) - Diagonal detail coefficients: captures high-frequency, fine details in the diagonal direction.
        ssim_ll = ssim(LL1, LL2, full=True, data_range=LL1.max() - LL1.min(), win_size=win_size)
        ssim_lh = ssim(LH1, LH2, full=True, data_range=LH1.max() - LH1.min(), win_size=win_size)
        ssim_hl = ssim(HL1, HL2, full=True, data_range=HL1.max() - HL1.min(), win_size=win_size)
        ssim_hh = ssim(HH1, HH2, full=True, data_range=HH1.max() - HH1.min(), win_size=win_size)

        # Return the mean SSIM across all coefficients (or you can use another aggregation method)
        ll_ssim = ssim_ll[0]
        lh_ssim = ssim_lh[0]
        hl_ssim = ssim_hl[0]
        hh_ssim = ssim_hh[0]
        cwssim_value = np.mean([ll_ssim, lh_ssim, hl_ssim, hh_ssim])
        return cwssim_value, ll_ssim, lh_ssim, hl_ssim, hh_ssim


    def compute_fsim(self):
        roi_image, roi_gt = self.extract_rois()

        # Convert both to grayscale
        if roi_image.ndim == 3:
            roi_image = color.rgb2gray(roi_image)
        if roi_gt.ndim == 3:
            roi_gt = color.rgb2gray(roi_gt)

        val = compute_fsim(roi_image, roi_gt)
        messagebox.showinfo("FSIM", f"FSIM: {val:.4f}")



    def compute_snr_and_display(self):
        roi_image, roi_gt = self.extract_rois()
        if self.color_mode == "Grayscale":
            roi_image = color.rgb2gray(roi_image)
            roi_gt = color.rgb2gray(roi_gt)
        psnr_value = calculate_psnr(roi_image, roi_gt)[0]
        messagebox.showinfo("PSNR", f"PSNR: {psnr_value:.2f} dB")

    def compute_ssim_and_display(self):
        roi_image, roi_gt = self.extract_rois()
        if self.color_mode == "Grayscale":
            roi_image = color.rgb2gray(roi_image)
            roi_gt = color.rgb2gray(roi_gt)
        ssim_index = ssim(roi_image, roi_gt, data_range=roi_image.max() - roi_image.min(), channel_axis=-1 if self.color_mode == "RGB" else None)
        messagebox.showinfo("SSIM", f"SSIM: {ssim_index:.4f}")

    def compute_mbsnr(self):
        roi_image, roi_gt = self.extract_rois()
        if self.color_mode == "Grayscale":
            roi_image = color.rgb2gray(roi_image)
            roi_gt = color.rgb2gray(roi_gt)
        val, signal, noise = calculate_mbsnr(roi_image, roi_gt)
        messagebox.showinfo("MBSNR", f"MBSNR: {val:.2f} dB\nSignal: {signal:.4f}\nNoise: {noise:.4f}")

    def compute_pbsnr(self):
        roi_image, roi_gt = self.extract_rois()
        if self.color_mode == "Grayscale":
            roi_image = color.rgb2gray(roi_image)
            roi_gt = color.rgb2gray(roi_gt)
        val, signal, noise = calculate_pbsnr(roi_image, roi_gt)
        messagebox.showinfo("PBSNR", f"PBSNR: {val:.2f} dB\nPower: {signal:.4f}\nNoise Power: {noise:.4f}")

    def compute_cwssim(self):
        roi_image, roi_gt = self.extract_rois()
        if self.color_mode == "Grayscale":
            roi_image = color.rgb2gray(roi_image)
            roi_gt = color.rgb2gray(roi_gt)
        val, ll, lh, hl, hh = calculate_cwssim(roi_image, roi_gt)
        messagebox.showinfo("CW-SSIM", f"CW-SSIM: {val:.4f}\nLL: {ll:.4f}\nLH: {lh:.4f}\nHL: {hl:.4f}\nHH: {hh:.4f}")
    

# Metric Functions

def calculate_mbsnr(roi_image, roi_gt):
    signal = np.mean(roi_gt)
    noise = np.std(roi_image - roi_gt)
    return 10 * np.log10(signal / noise) if noise > 0 else np.inf, signal, noise

def calculate_pbsnr(roi_image, roi_gt):
    y, x = np.asarray(roi_gt, dtype=np.float64), np.asarray(roi_image, dtype=np.float64)
    signal_power = np.sum(y ** 2)
    noise_power = np.sum((y - x) ** 2)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf, signal_power, noise_power

def calculate_psnr(x, y, max_pixel_value=1):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mse = np.mean((x - y) ** 2)
    return (10 * np.log10((max_pixel_value ** 2) / mse) if mse > 0 else float('inf'), (max_pixel_value**2), mse)

def complex_wavelet_transform(image):
    LL, (LH, HL, HH) = pywt.dwt2(image, 'bior1.3')
    return LL, LH, HL, HH

def calculate_cwssim(img1, img2):
    LL1, LH1, HL1, HH1 = complex_wavelet_transform(img1)
    LL2, LH2, HL2, HH2 = complex_wavelet_transform(img2)
    win_size = min(7, min(LL1.shape))
    if win_size % 2 == 0:
        win_size = 3
    ssim_ll = ssim(LL1, LL2, full=True, data_range=LL1.max() - LL1.min(), win_size=win_size)[0]
    ssim_lh = ssim(LH1, LH2, full=True, data_range=LH1.max() - LH1.min(), win_size=win_size)[0]
    ssim_hl = ssim(HL1, HL2, full=True, data_range=HL1.max() - HL1.min(), win_size=win_size)[0]
    ssim_hh = ssim(HH1, HH2, full=True, data_range=HH1.max() - HH1.min(), win_size=win_size)[0]
    return np.mean([ssim_ll, ssim_lh, ssim_hl, ssim_hh]), ssim_ll, ssim_lh, ssim_hl, ssim_hh

def compute_fsim(img1, img2):
    pc1 = phasecong(img1)[0]
    pc2 = phasecong(img2)[0]
    gm1 = np.sqrt(cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)**2 + cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)**2)
    gm2 = np.sqrt(cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)**2 + cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)**2)
    pc_sim = (2 * pc1 * pc2 + 1e-6) / (pc1**2 + pc2**2 + 1e-6)
    gm_sim = (2 * gm1 * gm2 + 1e-6) / (gm1**2 + gm2**2 + 1e-6)
    sim_map = gm_sim * pc_sim
    weighted_sim = sim_map * np.maximum(pc1, pc2)
    return np.sum(weighted_sim) / np.sum(np.maximum(pc1, pc2))


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSNRApp(root)
    root.mainloop()
