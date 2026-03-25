"""
This code is executed and tested on colab T4 GPU with 12.7 GB Ram and 15 GB GPU VRAM
"""

import os
import torch
import easyocr
import pandas as pd
import warnings
import re
from jiwer import cer, wer
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

warnings.filterwarnings('ignore')

Image.MAX_IMAGE_PIXELS = None  # Preventing Pillow Decompression Crash

# CONFIGURATION
TEST_CSV_PATH = "/content/drive/MyDrive/gsoc_2026/human_ai_assignment/specific_dataset/transcriptions.csv"
TEST_IMG_DIR = "/content/drive/MyDrive/gsoc_2026/human_ai_assignment/specific_dataset/images/"

# Directory to save the output transcriptions
OUTPUT_DIR = "./output_transcriptions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Qwen 2.5-VL 3B Instruct model
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Loading Models
print("Loading Stage 1: Convolutional-Recurrent Model (EasyOCR)...")
reader = easyocr.Reader(['es'], gpu=torch.cuda.is_available())

print(f"Loading Stage 2: Vision-Language Model ({VLM_MODEL_ID} in 4-bit)...")
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

vlm_model = AutoModelForImageTextToText.from_pretrained(
    VLM_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto"
)
vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)

# 2. Helper Functions
def normalize_text(text):
    """
    Removes newlines, double spaces, and punctuation.
    It does not lowercase, allowing Jiwer to penalize case errors and maintain originality from text.
    """
    text = str(text).replace('\n', ' ').replace('\r', '')
    # Added A-Z and uppercase Spanish accents
    text = re.sub(r'[^a-zA-Z0-9\sñáéíóúçÑÁÉÍÓÚÇ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def correct_with_vlm(image_path, raw_ocr_text):
    """
    Make the VLM to read the image, using the raw OCR as a structural hint,
    It is guided by a Few-Shot instructional prompt with strict language constraint. The prompt is improved or tuned iteratively to enhance the performance as possible.
    """
    system_prompt = (
        "You are an expert paleographer transcribing 18th-century Spanish texts. "
        "You will receive an image of the document and a raw, noisy OCR extraction hint. "
        "Your task: Output the PERFECT, exact transcription of the image. "
        "\n\nCRITICAL RULES:"
        "\n1. LANGUAGE LOCK: Output STRICTLY in Spanish using the Latin alphabet. ABSOLUTELY NO Cyrillic, Russian, or translations. Do not change the language."
        "\n2. COMPLETENESS: You MUST transcribe the ENTIRE text from start to finish. DO NOT stop early. DO NOT summarize."
        "\n3. DO NOT modernize the text. Preserve historical spellings exactly as printed."
        "\n4. Preserve 'long s' (ſ) if it appears, do not confuse it with 'f'."
        "\n5. Preserve 'u' and 'v' exactly as printed (e.g., write 'vno', not 'uno' if printed that way)."
        "\n6. The old spelling 'ç' must ALWAYS be written as 'z' (e.g., 'cobrança' becomes 'cobranza')."
        "\n7. Preserve the exact line breaks as seen in the image."
        "\n8. Output ONLY the raw transcription text. No introductions, no explanations."
        "\n\nEXAMPLES OF EXPECTED BEHAVIOR:"
        "\nRaw OCR Hint: 'la cafa de fu Mageftad y cl efcudo'"
        "\nYour Output: 'la casa de su Magestad y el escudo'"
        "\n\nRaw OCR Hint: 'cobrança de vnos marauedis'"
        "\nYour Output: 'cobranza de unos maravedis'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image_path, "max_pixels": 1024 * 1024},
            {"type": "text", "text": f"Raw OCR Hint:\n{raw_ocr_text}\n\nNow, provide the exact transcription strictly from the image:"}
        ]}
    ]

    text_prompt = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = vlm_processor(
        text=[text_prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,     # Greedy decoding (It improved performance compared to sampling from distribution, mostly because the task was straight forward transcription improvement)
        )

    trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    generated_text = vlm_processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    return generated_text.strip()

# Evaluation Execution
if __name__ == "__main__":
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"\nEvaluating {len(test_df)} test images...\n")
    print("=" * 80)

    crnn_cers, crnn_wers = [], []
    vlm_cers, vlm_wers = [], []

    for index, row in test_df.iterrows():
        img_name = row['file_name']
        img_path = os.path.join(TEST_IMG_DIR, img_name)

        raw_gt = str(row['text']).replace("END OF EXTRACT", "")
        ground_truth = normalize_text(raw_gt)

        print(f"Processing: {img_name}")

        # --- Stage 1: Raw CRNN ---
        raw_results = reader.readtext(img_path, detail=0, paragraph=True)
        # Joining with newlines to preserve line-wise structure
        raw_crnn_text_linewise = "\n".join(raw_results) 
        
        # Normalizes for evaluation math
        raw_crnn_text = normalize_text(raw_crnn_text_linewise)

        crnn_cer = cer(ground_truth, raw_crnn_text)
        crnn_wer = wer(ground_truth, raw_crnn_text)
        crnn_cers.append(crnn_cer)
        crnn_wers.append(crnn_wer)

        # --- Stage 2: VLM Visually Grounded Correction ---
        # Passes the text to the VLM so it understands the formatting context
        vlm_raw_output = correct_with_vlm(img_path, raw_crnn_text_linewise)
        
        # Normalizes final output purely for fair grading
        refined_text = normalize_text(vlm_raw_output)

        # Uses Character Error Rate (CER) & Word Error Rate (WER) as key evaluation metric
        final_cer = cer(ground_truth, refined_text)
        final_wer = wer(ground_truth, refined_text)
        vlm_cers.append(final_cer)
        vlm_wers.append(final_wer)
        
        # File Saving
        base_name = os.path.splitext(img_name)[0]
        
        with open(os.path.join(OUTPUT_DIR, f"{base_name}_EasyOCR.txt"), "w", encoding="utf-8") as f:
            f.write(raw_crnn_text_linewise)
            
        with open(os.path.join(OUTPUT_DIR, f"{base_name}_VLM.txt"), "w", encoding="utf-8") as f:
            f.write(vlm_raw_output)

        # Print Side-by-Side Comparison (Truncated for terminal readability)
        print(f"Ground Truth : {ground_truth[:120]}...")
        print(f"Raw EasyOCR  : {raw_crnn_text[:120]}... | CER: {crnn_cer:.4f}")
        print(f"With VLM Correction  : {refined_text[:120]}... | CER: {final_cer:.4f}")
        print("-" * 80)

    print(f"\nAll line-wise transcripts saved successfully to: {OUTPUT_DIR}")

    print("\nFINAL TWO-STAGE PIPELINE METRICS")
    print("| Pipeline Stage                  | Avg CER | Avg WER |")
    print("|---------------------------------|---------|---------|")
    print(f"| 1. Base CRNN (EasyOCR)          | {sum(crnn_cers)/len(crnn_cers):.4f}  | {sum(crnn_wers)/len(crnn_wers):.4f}  |")
    print(f"| 2. EasyOCR + Qwen 2.5-VL | {sum(vlm_cers)/len(vlm_cers):.4f}  | {sum(vlm_wers)/len(vlm_wers):.4f}  |")