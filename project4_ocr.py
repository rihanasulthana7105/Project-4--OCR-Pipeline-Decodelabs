"""
Project 4: Optical Character Recognition (OCR) pipeline
Pipeline: Load -> Grayscale -> Gaussian Blur -> Adaptive Threshold
          -> Tesseract OCR -> Confidence filter (>= 80%) -> Visual output

Install:
    pip install opencv-python pytesseract numpy
    Also install the Tesseract engine itself:
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      Linux:   sudo apt install tesseract-ocr
      macOS:   brew install tesseract
Run:
    python project4_ocr.py your_image.jpg
"""

import sys
import cv2
import numpy as np
import pytesseract

# Windows only: uncomment and fix the path if Tesseract is not on PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CONF_THRESHOLD = 80   # minimum confidence (%) to keep a word
PSM_MODE = 6          # 3=auto, 6=text block, 7=single line, 11=sparse text


def preprocess(image):
    """Grayscale -> Gaussian blur -> adaptive threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )
    return gray, blur, thresh


def deskew(binary):
    """Optional: straighten tilted text. Returns the corrected image."""
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) < 50:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return binary
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(binary, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def run_ocr(binary):
    """Run Tesseract and keep only words with confidence >= threshold."""
    config = f"--psm {PSM_MODE}"
    data = pytesseract.image_to_data(
        binary, config=config, output_type=pytesseract.Output.DICT
    )
    results = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = float(data["conf"][i])
        if text and conf >= CONF_THRESHOLD:   # the 80% gate
            results.append({
                "text": text,
                "conf": conf,
                "box": (data["left"][i], data["top"][i],
                        data["width"][i], data["height"][i]),
            })
    return results


def draw_results(image, results):
    out = image.copy()
    for r in results:
        x, y, w, h = r["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(out, f'{r["text"]} {r["conf"]:.0f}%', (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python project4_ocr.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1])
    if image is None:
        print("Could not read image. Check the path.")
        sys.exit(1)

    gray, blur, thresh = preprocess(image)
    thresh = deskew(thresh)          # remove this line if text is already straight
    results = run_ocr(thresh)

    print("\n=== OCR RESULT (confidence >= 80%) ===")
    print(" ".join(r["text"] for r in results))

    if results:
        avg = sum(r["conf"] for r in results) / len(results)
        print(f"\nWords kept: {len(results)}")
        print(f"Average confidence: {avg:.1f}%")
        print("PASS: validated confidence >= 80%" if avg >= CONF_THRESHOLD
              else "FAIL: try a clearer image or another PSM mode")
    else:
        print("No words passed the 80% threshold. Try a sharper image or change PSM_MODE.")

    # Save visual proof of each pipeline step
    cv2.imwrite("1_grayscale.png", gray)
    cv2.imwrite("2_blur.png", blur)
    cv2.imwrite("3_adaptive_threshold.png", thresh)
    cv2.imwrite("4_final_output.png", draw_results(image, results))
    print("\nSaved: 1_grayscale.png, 2_blur.png, 3_adaptive_threshold.png, 4_final_output.png")


if __name__ == "__main__":
    main()
