import os
import re
import cv2
import numpy as np
import pandas as pd
from docx import Document
from pdf2image import convert_from_path
from PIL import Image 

Image.MAX_IMAGE_PIXELS = None

# CONFIGURATION
PDF_DIR = "/content/drive/MyDrive/gsoc_2026/human_ai_assignment/Test_sources/Print"
DOCX_DIR = "/content/drive/MyDrive/gsoc_2026/human_ai_assignment/Test_transcriptions/Print"
OUTPUT_IMG_DIR = "/content/drive/MyDrive/gsoc_2026/human_ai_assignment/specific_dataset2/images/"
OUTPUT_CSV = "/content/drive/MyDrive/gsoc_2026/human_ai_assignment/specific_dataset2/transcriptions.csv"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# HELPER: Parses Word Doc (Page Level)
def parse_transcription_docx_page_level(docx_path):
    """
    Groups all text lines on a page into a single multi-line string.
    """
    doc = Document(docx_path)
    page_texts = {}
    current_page = None
    text_buffer = []

    page_marker_regex = re.compile(r"PDF\s*p\.?\s*(\d+)", re.IGNORECASE)

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        match = page_marker_regex.match(text)
        if match:
            if current_page is not None and text_buffer:
                page_texts[current_page] = "\n".join(text_buffer)
            current_page = int(match.group(1))
            text_buffer = []
        else:
            if current_page is not None and not text.startswith("NOTES:"):
                text_buffer.append(text)

    if current_page is not None and text_buffer:
        page_texts[current_page] = "\n".join(text_buffer)

    return page_texts

# HELPER: OPENCV Column Cropper
def crop_main_text_column(img_cv):
    """
    Crops out the marginalia, leaving just the main text body.
    """
    height, width = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_x_mins, valid_y_mins, valid_x_maxs, valid_y_maxs = [], [], [], []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_x = x + (w / 2)

        # Ignore far right 20% or far left 15% (Marginalia)
        if center_x > (width * 0.80) or center_x < (width * 0.15):
            continue
        if w < 50 or h < 50:
            continue

        valid_x_mins.append(x)
        valid_y_mins.append(y)
        valid_x_maxs.append(x + w)
        valid_y_maxs.append(y + h)

    if not valid_x_mins:
        return img_cv

    pad = 15
    x1 = max(0, min(valid_x_mins) - pad)
    y1 = max(0, min(valid_y_mins) - pad)
    x2 = min(width, max(valid_x_maxs) + pad)
    y2 = min(height, max(valid_y_maxs) + pad)

    return img_cv[y1:y2, x1:x2]

# Main Execution
if __name__ == "__main__":
    csv_data = []

    for pdf_filename in os.listdir(PDF_DIR):
        if not pdf_filename.endswith(".pdf"): continue

        base_name = os.path.splitext(pdf_filename)[0]
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        docx_path = os.path.join(DOCX_DIR, f"{base_name}.docx")

        if not os.path.exists(docx_path):
            print(f"Skipping {pdf_filename}: No matching docx found.")
            continue

        print(f"\nProcessing Pair: {pdf_filename} & {base_name}.docx")
        page_texts = parse_transcription_docx_page_level(docx_path)

        for page_num, full_page_text in page_texts.items():
            try:
                images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)
                if not images: continue

                img_cv = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)

                # Crops the main column
                cropped_img = crop_main_text_column(img_cv)

                # Saves the whole column as one image
                img_filename = f"{base_name}_page_{page_num}.jpg"
                img_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
                cv2.imwrite(img_path, cropped_img)

                # Pairs it with the giant text block
                csv_data.append({
                    "file_name": img_filename,
                    "text": full_page_text
                })
                print(f"  -> Extracted Page {page_num} successfully.")

            except Exception as e:
                print(f"Error on {pdf_path} Page {page_num}: {e}")

    df = pd.DataFrame(csv_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDataset build complete! Extracted {len(df)} page-level pairs.")