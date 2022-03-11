from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\sukru.alkan\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"

img = Image.open("text.png")
text = pytesseract.image_to_string(img, lang="eng")
print(text)
