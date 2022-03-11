#araç plakası okuma
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\sukru.alkan\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"
import imutils

img = cv2.imread("licence_plate.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 6, 250, 250) #gri resim, diameter(çap), sigma color, sigma space
#burada amaç plaka dışındaki yerleri filtrelemeyle keskin çizgilerini azaltarak gürültü gibi kabul etmek neden yapıyoruz kenar sayısını azaltmak için
edged = cv2.Canny(filtered, 30, 200) #kenarları algıladık

contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #sınır çizgilerin koordinatlarını buluyoruz
cnts = imutils.grab_contours(contours) #uygun(düzgün) kontur(kenar) degerlerini yakala
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10] #bulduğum konturları alanlarına göre sıraladım çünkü dikdörtgen arıyorum(0 dan 10 a kadar olanları)
screen = None

#dört kenatlı kapalı bir şekil bulmak için yapıyoruz
for c in cnts:
    epsilon = 0.018*cv2.arcLength(c, True) #dış görünüşleri bozuk çokgenleri tespit etmek için kullanılan deneysel formül
    approx = cv2.approxPolyDP(c, epsilon, True) #hatalı sınır algılamalarını en aza indirip çokgen koordinatlarını testip ediyor
    if len(approx) == 4: #4 köşe koordinatı varsa
        screen = approx
        break

mask = np.zeros(gray.shape, np.uint8) #resimle aynı boyutta siyah bir pencere yaptık
new_img = cv2.drawContours(mask, [screen], 0, (255, 255, 255), -1) #plakanın olduğu yeri beyaz yapacağız
new_img = cv2.bitwise_and(img, img, mask=mask) #plaka bölgesindeki yazıyı o bölgeye papıştırmak

#plakanın olduğu yeri alıp diğer yerleri beyaza boyadı
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]

text = pytesseract.image_to_string(cropped, lang="eng") #plakayı okuma
print("deteced text", text)

cv2.imshow("resim", img)
cv2.imshow("plaka", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()