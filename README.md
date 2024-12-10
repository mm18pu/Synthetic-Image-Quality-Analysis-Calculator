# Synthetic Image Quality Calculator

The **Synthetic Image Quality Calculator** is a versatile application designed for assessing the quality of images based on various metrics. The tool allows you to calculate metrics such as Signal-to-Noise Ratio (SNR) and Structural Similarity Index Measure (SSIM) for grayscale and RGB images, including detailed SSIM components like luminance, contrast, and structure.

This application has been tested and verified to run on **Linux**, **Windows**, and **Mac** operating systems.

---

## Features

# Synthetic Image Quality Calculator

This application provides a graphical interface for evaluating the quality of synthetic images compared to a ground truth image. It calculates Signal-to-Noise Ratio (SNR) and Structural Similarity Index (SSIM) for selected Regions of Interest (ROIs). The tool supports both grayscale and RGB color modes.

## Features

- **Load Images**: Load the synthetic image and ground truth image.
- **Toggle Color Mode**: Switch between RGB and grayscale for analysis.
- **Calculate SNR**: Compute the SNR in decibels (dB) for a selected ROI.
- **Calculate SSIM**: Compute the SSIM for a selected ROI.
- **Save Image & ROI**: Save the ROI data and selected image regions for later analysis.
- **Batch Processing**: Automate SNR and SSIM calculations for multiple image pairs with ROI data from a CSV file.
- **Output CSV**: Generate a CSV file with detailed calculations:
  - SNR (Grayscale, RGB)
  - SSIM (Grayscale, RGB)
  - SSIM components (Luminance, Contrast, Structure) for grayscale and individual R, G, B
---

## Requirements

Before running the application, ensure that the following dependencies are installed:

- Python 3.8 or higher
- Required Python libraries:
  - `numpy`
  - `opencv-python`
  - `scikit-image`
  - `pandas`
  - `matplotlib`
  - `csv`
  - tkinter
  - `PIL`
  - `scipy`

Install the dependencies using the following command:

```bash
pip install numpy opencv-python scikit-image pandas matplotlib
```

## GUI Mode

1. **Run the application**: Start the GUI by executing the following command in your terminal:
   ```bash
   python synthetic_image_quality_calculator.py
   ```
2. **Load Images:** Use the "Load Image" buttons in the GUI to load both the synthetic image and the ground truth image. Each image will appear in its respective display area.

3. **Select an ROI:** Click and drag on either image to select a Region of Interest (ROI). The ROI will be highlighted, and its coordinates will be used for subsequent calculations.

4. **Toggle Between Color Modes:** Use the "Grayscale" or "RGB" toggle options to switch between grayscale and RGB color modes for analysis.

5. **Calculate SNR:** After selecting an ROI, click the "Calculate SNR" button to compute the Signal-to-Noise Ratio (SNR) for the selected region. The result will display in the GUI.

6. **Calculate SSIM:** Click the "Calculate SSIM" button to calculate the Structural Similarity Index (SSIM) for the ROI. Results include SSIM scores for grayscale or RGB, depending on the current mode.

7. **Save Results and ROI Data:** Use the "Save" button to store the ROI data and calculated metrics. Saved ROIs can be reused for later analysis.

8. **Close the Application:** Once finished, exit the application by clicking the "Close" button or closing the GUI window.




