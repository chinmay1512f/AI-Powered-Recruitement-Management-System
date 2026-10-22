import fitz
import pytesseract
from PIL import Image
import docx

# Safety: works even if PATH fails
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(file_path):
    text = ""

    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
            else:
                pix = page.get_pixmap()
                img = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples
                )
                text += pytesseract.image_to_string(img)

    elif file_path.lower().endswith(".docx"):
        document = docx.Document(file_path)
        for para in document.paragraphs:
            text += para.text + "\n"

    elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file_path)
        text += pytesseract.image_to_string(img)

    return text
