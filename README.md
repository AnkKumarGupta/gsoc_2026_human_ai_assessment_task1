# GSoC 2026 Human AI Assessment: RenAIssance Project

**Project Title:** Automating text recognition and transcription of historical documents with weighted convolutional - recurrent architectures and LLM integration.

## Task 1: Optical Character Recognition of Printed Sources

This repository contains the solution and experimental methodology for Task 1 of the GSoC 2026 assessment. The goal is to accurately transcribe 18th-century Spanish historical documents. Some of the key challenges werenavigating challenges were obsolete typography (e.g., the long 's' (ſ)), interchangeable 'u'/'v' usage, faded ink, and human-transcriber editorial conventions.

---

## Ground Truth Data Preparation (`dataset_preparation.py`)


Before evaluating any models, it was critical to construct a pristine, 1:1 ground truth dataset from the provided raw PDFs and transcriber Word documents. If the evaluation dataset contains formatting artifacts or transcriber typos, the final CER/WER metrics will be artificially penalized. 

The dataset preparation was executed in a two-step process:

**1. Automated Parsing and Cropping:**
I engineered a Python script (`dataset_preparation.py`) to systematically build the dataset:
* **Text Extraction:** Parsed the provided `.docx` files using regex (`PDF p.\d+`) to isolate page-level transcriptions while programmatically ignoring transcriber metadata (like "NOTES:").
* **High-Fidelity Rendering:** Converted the corresponding `.pdf` pages into 300 DPI images.
* **OpenCV Spatial Isolation:** Applied OpenCV thresholding and morphological dilation to detect text contours. This allowed the script to dynamically crop out the physical margins (ignoring the far right 20% and left 15%), effectively removing library stamps and marginalia noise so the OCR models could focus purely on the main text body.

**2. Manual Verification and Correction:**
While the script handled the bulk parsing, human transcriptions often contain slight misalignments or missed characters. I manually verified the generated `transcriptions.csv` against every single cropped image. I corrected missed characters, ensured the line breaks matched the physical prints, in some of the transcriptions, left and right page transcriptions were separate, there was also few errors, corrected them and guaranteed that the final **24 test images** along with a **transcription.csv** file that possessed a flawless, character-perfect Ground Truth.

---

### Final Architecture: The Rule-Aligned Hybrid Pipeline

The final submitted solution utilizes a two-stage hybrid architecture designed to maximize accuracy while fitting within the constraints of consumer-grade hardware (15GB VRAM, Google Colab T4):

1. **Stage 1: Base Structural Extraction (EasyOCR)**
   A pretrained Convolutional-Recurrent Neural Network (CRNN) is used zero-shot to extract the initial draft transcription text. 
2. **Stage 2: Visually-Grounded Proofreading (Qwen 2.5-VL-3B-Instruct)**
   A 3-Billion parameter Vision-Language Model, quantized in 4-bit, that takes both the original document image and the EasyOCR draft as inputs. It acts as an expert paleographer, proofreading the draft against the visual evidence and applying strict editorial rules to produce the final transcription.

---

## The Engineering Journey: Iterations & Findings

Building this pipeline required navigating several classic machine learning failure modes. The final architecture is the result of rigorous ablation testing.

### Iteration 1: The Base CRNN (EasyOCR)
* **Approach:** Ran standard EasyOCR over the dataset.
* **Result:** Reached an average Character Error Rate (CER) of ~0.28. 
* **Failure Point:** The model successfully captured the document structure but failed entirely on 18th-century orthography, frequently confusing 'f' with the long 's' (ſ) and struggling with faded characters.

### Iteration 2: Text-Only LLM Late-Stage Correction
* **Approach:** Attempted to use a text-only LLM (Qwen 3B) fine-tuned via LoRA to map the noisy EasyOCR output to the clean ground truth.
* **Result:** CER increased. 
* **Failure Point:** **Modern Language Drift and Subword Hallucination.** Mostly because the text LLM could not "see" the original image, it relied purely on its linguistic prior. When presented with OCR noise (e.g., `difleño`), instead of correcting individual characters, it hallucinated modern Spanish equivalents (e.g., rewriting it as `distinguida`), destroying the historical accuracy.

### Iteration 3: Zero-Shot VLM Processing 
* **Approach:** Passed the image and the OCR draft to a Vision-Language Model to visually ground the text correction.
* **Result:** Encountered VRAM crashes, followed by severe hallucination loops.
* **Failure Point:** **Activation Memory and Anchoring Bias.** Processing high-resolution documents scales quadratically in Transformer attention layers ($O(N^2)$). To prevent CUDA Out-of-Memory crashes, `max_pixels` was capped at 1024x1024. But this can make the image blurry. The VLM lost confidence, fell into autoregressive stuttering loops (e.g., outputting "9 9 9 9" some times), and occasionally hallucinated Cyrillic translations because it couldn't resolve the Latin characters clearly.

### Iteration 4 (Final): Prompt Straitjacketing and Rule Alignment
* **Approach:** As the Ground Truth dataset contained implicit human editorial expansions (e.g., expanding a capped 'q' to 'que', and standardizing the old 'ç' to 'z'). Rewrote the VLM System Prompt into a programmable rule engine. Added a `LANGUAGE LOCK` to prevent Cyrillic hallucinations, applied a `repetition_penalty` to stop stuttering, and explicitly fed the human editorial rules into the prompt context. 
* **Final Result:** The VLM successfully utilized the EasyOCR draft for structure, applied the historical rules, and dropped the average CER significantly compared to the baseline, achieving sub-0.10 CER on high-quality pages.

---

## Results and Performance Analysis



The final two-stage pipeline was evaluated on a the ground truth test set of 24 document images. The results demonstrated a clear improvement when integrating the Vision-Language Model as a late-stage reasoning and correction layer.

### Final Pipeline Metrics
| Pipeline Stage                  | Avg CER | Avg WER |
|---------------------------------|---------|---------|
| 1. Base CRNN (EasyOCR)          | 0.2855  | 0.7621  |
| 2. EasyOCR + Qwen 2.5-VL        | **0.2310** | **0.4309** |

By utilizing the Qwen 2.5-VL model with Few-Shot prompting and strict language constraints, **the Character Error Rate (CER) was reduced by ~19%, and the Word Error Rate (WER) was slashed by over 43%.** 

### Best vs. Worst Performance

**The Best Performances (VLM CER < 0.06):**
Where the VLM succeeded, it achieved near-perfect historical transcription. 
* *Guardiola - Tratado nobleza_page_14.jpg:* VLM CER: **0.0394** (Improved from EasyOCR's 0.2348)
* *Guardiola - Tratado nobleza_page_13.jpg:* VLM CER: **0.0486** (Improved from EasyOCR's 0.2538)
* *Covarrubias - Tesoro lengua_page_7.jpg:* VLM CER: **0.0547** (Improved from EasyOCR's 0.1743)

**The Worst Performances (VLM CER > 0.90):**
These highest error rates were mostly caused by hardware-induced early truncation.
* *PORCONES.23.5 - 1628_page_2_left.jpg:* VLM CER: **0.9478** * *PORCONES.748.6 - 1650_page_4.jpg:* VLM CER: **0.9027**, although easyocr CER were **0.2381** and **0.2018** respectively, which is very much comparable to other easyOCR results.
* **Why it failed:** The performance deteriorated after VLM correction mostly because to prevent the Colab GPU (15GB VRAM) from crashing due to $O(N^2)$ attention memory limits, the images had to be scaled down to 1024x1024 pixels. For extremely dense pages, the text became too blurry. The model lost visual confidence and either entered a repetitive token loop (e.g., `ocos ocos ocos`) or triggered an early `<eos>` token, stopping the transcription after just a few words. 

### Transcription Example & Analysis

To understand *how* the VLM improves the text, this excerpt from `Guardiola - Tratado nobleza_page_14.jpg` can be observed:

* **Ground Truth:** `caer e incurrir en las penas contenidas en la dicha pragmatica e leyes de nuestros Reynos`
* **Raw EasyOCR:** `cacr c incurrir cn las pcnas contcnidas cn la dicha PIagmatica c lcyes de nucfros Reynos`
* **Expert VLM:** `caer é incurrir en las penas contenidas en la dicha pragmatica e leyes de nuestros Reynos`

**Analysis:** EasyOCR struggles with the physical artifacts of the 18th-century printing press, constantly misinterpreting 'e' as 'c' (e.g., *cacr*, *pcnas*, *lcyes*) because of faded ink loops. It also hallucinated capital letters (*PIagmatica*). The VLM, grounded by both the visual image and its internal linguistic model, seamlessly reconstructs the actual Spanish words, applies proper spacing, and correctly identifies the lowercase letters, proving its immense value as a paleographic proofreader.

---

## Evaluation Metrics: CER & WER



To rigorously evaluate transcription accuracy, this pipeline utilizes **Character Error Rate (CER)** and **Word Error Rate (WER)**. Few other advanced metrics might be BLEU or ROUGE, but In historical Document AI like this one, standard NLP metrics like BLEU or ROUGE are inappropriate because they measure semantic meaning and translation quality. Historical transcription requires exact, literal string matching.

**1. Character Error Rate (CER)**
CER is based on the Levenshtein distance and operates at the character level. It calculates the minimum number of character-level operations required to transform the model's prediction into the exact ground truth.
* **Formula:** `CER = (Substitutions + Insertions + Deletions) / Total Characters in Ground Truth`
* **Why it is crucial here:** In 18th-century Spanish, the difference between a long 's' (ſ) and an 'f' is a single character. CER is highly sensitive to these micro-level typographical errors, making it the ultimate benchmark for paleographic fidelity.

**2. Word Error Rate (WER)**
WER operates on the exact same Levenshtein distance principle but calculates substitutions, insertions, and deletions at the *word* level.
* **Why it is used alongside CER:** A model might have a decent CER but a high WER if it struggles with spacing (e.g., concatenating "de el" into "deel") or hallucinates random characters at the start of a page. The 43% drop in WER in Stage 2 proves that the VLM is exceptionally good at fixing EasyOCR's spatial and bounding-box grouping errors, restoring the true structural readability of the text.

**The Normalization Strategy:**
To ensure the metrics graded the model's actual paleographic accuracy rather than arbitrary formatting, a strict normalization function was applied prior to evaluation. The function preserved exact capitalization and historical Spanish characters (ñ, á, é, í, ó, ú, ç) but stripped all subjective punctuation (commas, periods) and standardized whitespaces. This guaranteed a fair, mathematically rigorous comparison.

---

## Limitations

* **Hardware Bottlenecks:** The pipeline's accuracy is artificially limited by the 15GB VRAM ceiling. Capping `max_pixels` at 1024x1024 degrades visual fidelity, forcing the VLM to rely heavier on the EasyOCR draft than is ideal.
* **Anchoring Bias:** Because of the downsampling, the VLM sometimes defaults to the erroneous EasyOCR text when the pixels are too blurry to decode independently.
* **Lack of Large-Scale Fine-Tuning:** While in-context learning and prompt engineering worked for this PoC, it is not perfectly stable. An aligned bigger dataset can be found or built for fine tuning, it can improve the performance. 

---

## How to Run the Code

### Prerequisites
A CUDA-enabled GPU with at least 15GB of VRAM (e.g., Google Colab T4).

### Setup
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the code file `task1_code.py`