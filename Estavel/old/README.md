# Overview

This Python3 script extracts text and tables from PDF files using multiple selectable engines, saves the extracted content into a MongoDB collection, and creates a local JSON backup for each processed PDF.

### Supported Engines:

* **PyPDF2**: Basic text extraction.
* **PDFMiner.six**: Advanced layout control.
* **PyMuPDF (fitz)**: Fast text and image extraction.
* **Tabula-py**: Table extraction via Java.
* **Camelot**: Vision-based table detection.
* **OCR**: Tesseract‑based OCR extraction for scanned or image‑based PDFs (fallback engine).

### Key Features:

* **Multi-engine extraction**: Choose from six engines for different use cases.
* **Automatic dependency checks & fallback**: Detects missing binaries (e.g., Java for Tabula-py, Ghostscript for Camelot, Tesseract/Poppler for OCR) and switches automatically to a supported engine or OCR.
* **OCR fallback**: If the chosen engine fails or extracts fewer characters than the specified threshold, the script runs an OCR pass to ensure no content is missed.
* **Robust error handling**: Wraps engine calls in `try/except` and ensures text extraction always proceeds.
* **MongoDB storage**: Saves extracted content into a MongoDB collection with fields: `name`, `filename`, and `text`.
* **Local JSON backup**: Creates a readable `.json` file for each processed PDF.
* **Configurable via CLI**: Command-line arguments allow customization of PDF path, record name, engine choice, JSON output directory, and OCR threshold.

---

# Dependencies

### System Dependencies:

1. **Java Runtime Environment (JRE 8+)** (for Tabula-py):

   ```bash
   sudo apt update && sudo apt install -y default-jre
   ```
2. **Ghostscript** (for Camelot):

   ```bash
   sudo apt install -y ghostscript
   ```
3. **OpenCV and Tkinter** (optional for Camelot):

   ```bash
   sudo apt install -y python3-opencv python3-tk
   ```
4. **Tesseract OCR** (for OCR engine):

   ```bash
   sudo apt install -y tesseract-ocr libtesseract-dev
   ```
5. **Poppler Utilities** (for PDF→image conversion):

   ```bash
   sudo apt install -y poppler-utils
   ```

### Python Packages:

Install required libraries via pip:

```bash
pip install \
  pymongo PyPDF2 pdfminer.six PyMuPDF \
  tabula-py "camelot-py[base]" ghostscript \
  pdf2image pytesseract
```

* **pymongo**: MongoDB driver
* **PyPDF2**: Lightweight PDF reading/extraction
* **pdfminer.six**: Detailed layout-aware extraction
* **PyMuPDF**: Fast text/image extraction
* **tabula-py\[jpype]**: Table extraction via Java
* **camelot-py\[base]**: Vision-based table parsing
* **ghostscript**: Python wrapper for Ghostscript C library
* **pdf2image** / **pytesseract**: PDF→image conversion and OCR

---

# Installation

1. Clone or download the script to your local machine.
2. Ensure system dependencies (Java, Ghostscript, OpenCV/Tkinter, Tesseract, Poppler) are installed.
3. Install Python packages via pip (see Dependencies).
4. (Optional) Make the script executable:

   ```bash
   chmod +x pdf_to_mongo.py
   ```

---

# Usage

Run the script with the following syntax:

```bash
./pdf_to_mongo.py <PDF_PATH> "<RECORD_NAME>" [--engine ENGINE] [--json-dir DIR] [--ocr-threshold N]
```

### Arguments:

* `<PDF_PATH>`: Path to the PDF file.
* `<RECORD_NAME>`: Custom name for the record.
* `--engine ENGINE`: Extraction engine (`pypdf2`, `pdfminer`, `pymupdf`, `tabula`, `camelot`, `ocr`). Default: `pypdf2`.
* `--json-dir DIR`: Directory for JSON backup (default: `pdf_jsons`).
* `--ocr-threshold N`: Minimum characters before triggering OCR fallback (default: `100`).

### Examples:

1. **Basic text extraction** with PyPDF2 (default engine):

   ```bash
   python3 pdf_to_mongo.py caminho/para/arquivo.pdf "Nome do Registro"
   ```
2. **Forçar PyPDF2 explicitamente**:

   ```bash
   python3 pdf_to_mongo.py caminho/para/arquivo.pdf "Nome do Registro" --engine pypdf2 --json-dir dados_json (NÃO BOM)
   ```
3. **Extração avançada** com PDFMiner.six:

   ```bash
   python3 pdf_to_mongo.py caminho/para/arquivo.pdf "Meu Relatório Complexo" --engine pdfminer --json-dir dados_json (NÃO BOM)
   ```
4. **Extração rápida** de texto e imagens com PyMuPDF:

   ```bash
   python3 pdf_to_mongo.py caminho/para/arquivo.pdf "Apresentação" --engine pymupdf --json-dir dados_json (BOM)
   ```
5. **Extração de tabelas** via Tabula-py (requer Java):

   ```bash
   python3 pdf_to_mongo.py caminho/para/arquivo.pdf "Tabelas de Vendas" --engine tabula --json-dir dados_json (NÃO BOM)
   ```
6. **Detecção de tabelas via visão computacional** com Camelot (requer Ghostscript/OpenCV):

   ```bash
   python3 pdf_to_mongo.py caminho/para/arquivo.pdf "Tabelas Visionárias" --engine camelot --json-dir dados_json (NÃO BOM)
   ```
7. **Extração OCR completa** para PDFs escaneados com Tesseract:

   ```bash
   python3 pdf_to_mongo.py caminho/para/arquivo.pdf "Documento Escaneado" --engine ocr --ocr-threshold 0 --json-dir dados_json (BOM)
   ```
8. **Alterar diretório de saída JSON**:

   ```bash
   python3 pdf_to_mongo.py arquivo.pdf "BackupJSON" --json-dir meus_backups --json-dir dados_json
   ```
9. **Definir limiar mínimo para acionar OCR automático**:

   ```bash
   python3 pdf_to_mongo.py arquivo.pdf "ThresholdDemo" --engine pdfminer --ocr-threshold 200 --json-dir dados_json (NÃO BOM)
   ```
10. **Combinar engine, diretório JSON e threshold**:

    ```bash
    python3 pdf_to_mongo.py arquivo.pdf "FullConfig" --engine camelot --json-dir backups_json --ocr-threshold 50 (NÃO BOM)
    ```
11. **Forçar fallback OCR mesmo com extração prévia** (threshold = 0):

    ```bash
    python3 pdf_to_mongo.py arquivo.pdf "SóOCR" --engine pdfminer --ocr-threshold 0 --json-dir dados_json (NÃO BOM)
    ```

---

# Fallback Behavior

* If **Java** is missing, the script switches Tabula-py → PDFMiner.six.
* If **Ghostscript** is missing, the script switches Camelot → PDFMiner.six.
* If the chosen engine fails at runtime, the script falls back to OCR.
* If extracted text length < `--ocr-threshold`, an OCR pass is triggered.

---

# Contributing

Contributions, issues, and feature requests are welcome! Submit a pull request or open an issue on the project repository.

---

# License

This project is licensed under the MIT License.