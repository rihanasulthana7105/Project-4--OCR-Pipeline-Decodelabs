# Project 4: OCR Pipeline with Tesseract and OpenCV

A Python computer vision pipeline that takes a raw image, cleans it with image pre-processing, extracts text using OCR, and keeps only results with **at least 80% confidence**.

## Result

| Metric | Value |
|---|---|
| Average confidence | **95.7%** |
| Confidence threshold | 80% |
| Status | **PASS** |

![Final output](4_final_output.png)

## Pipeline

1. **Grayscale conversion:** collapses the 3-channel RGB image into a single intensity channel.
2. **Gaussian blur:** removes small noise and artifacts.
3. **Adaptive thresholding:** converts the image to pure black and white so characters stand out from the background.
4. **Deskewing:** straightens tilted text.
5. **OCR with Tesseract:** extracts words and per-word confidence scores using `pytesseract.image_to_data`.
6. **Confidence filter:** keeps a word only if `confidence >= 80`.
7. **Visual output:** draws a bounding box and label on every accepted word.

## Tech Stack

- Python
- OpenCV (`opencv-python`)
- Tesseract OCR via `pytesseract`
- NumPy, Matplotlib
- Google Colab

## Files

| File | Description |
|---|---|
| `Project4_OCR_Pipeline.ipynb` | Colab notebook with all steps and saved outputs |
| `project4_ocr.py` | Standalone script version |
| `sample_invoice.png` | Sample input image |
| `1_grayscale.png` | Output of grayscale step |
| `2_blur.png` | Output of Gaussian blur step |
| `3_adaptive_threshold.png` | Output of adaptive thresholding |
| `4_final_output.png` | Final image with bounding boxes and confidence labels |

## How to Run

### Option 1: Google Colab
1. Open `Project4_OCR_Pipeline.ipynb` in [Google Colab](https://colab.research.google.com).
2. Click **Runtime > Run all**.
3. Upload an image of printed text when asked.

### Option 2: Local
1. Install the Tesseract engine:
   - Windows: install from the UB-Mannheim Tesseract page
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`
2. Install the Python libraries:
   ```
   pip install opencv-python pytesseract numpy
   ```
3. Run:
   ```
   python project4_ocr.py sample_invoice.png
   ```

## Configuration

Change these values at the top of the script or notebook:

- `CONF_THRESHOLD = 80`: minimum confidence (%) to keep a word
- `PSM_MODE`: Tesseract page segmentation mode (`3` auto, `6` text block, `7` single line, `11` sparse text such as invoices)

## What I Learned

- How an image is represented as a numeric matrix of pixel values
- Why pre-processing (grayscale, blur, thresholding) directly improves OCR accuracy
- How confidence scores and thresholds reduce false positives
- Choosing the right Tesseract page segmentation mode for the layout

## Author

**I. Rihana Sulthana**  
AI Intern at DecodeLabs (Batch 2026)
