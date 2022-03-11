import cv2
import numpy as np
import math

vid = cv2.VideoCapture(0)


while 1:
    try:
        ret, frame = vid.read()
        frame = cv2.flip(frame, 1) #y ekseninde yansıtma(ayna görüntüsü)

        kernel = np.ones((3, 3), np.uint8)

        roi = frame[100:300, 100:300]
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 0)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], np.uint8)
        upper_skin = np.array([20, 255, 255], np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask, kernel, iterations=4) # mask işlemini yaparken oluşan siyah noktaları beyaz noktalar ile doldurduk
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key=lambda x: cv2.contourArea(x)) #contour alanlarını döndürür

        epsilon = 0.0005*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True) #contourlara daha iyi bir yaklaşım sağlar

        hull = cv2.convexHull(cnt)

        areaHull = cv2.contourArea(hull) #hull alanını tutacak
        areaCnt = cv2.contourArea(cnt) #elimizin alanını tutacak
        areaRatio = ((areaHull - areaCnt) / areaCnt) * 100 #elimizin olmadığı alan oranı

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull) #convexhull kusurlarını bul

        l = 0 #kusur sayısı
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            s = (a + b + c) / 2
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            d = (2 * area) / a

            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            cv2.line(roi, start, end, [255, 0, 0], 2)

        l += 1

        font = cv2.FONT_HERSHEY_SIMPLEX

        if l == 1:
            if areaCnt < 2000: #el roi dışında ise demek
                cv2.putText(frame, "Put your hand in the box!", (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if areaRatio < 12: #elimin olmadığı alanın yüzdesi 12 den küçükse
                    cv2.putText(frame, "0", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif areaRatio < 17.5:
                    cv2.putText(frame, "Best luck", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "1", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif l == 2:
            cv2.putText(frame, "2", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif l == 3:
            if areaRatio < 27:
                cv2.putText(frame, "3", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Ok", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif l == 4:
            cv2.putText(frame, "4", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif l == 6:
            cv2.putText(frame, "Reposition", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Reposition", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)

    except:
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

vid.release()
cv2.destroyAllWindows()
