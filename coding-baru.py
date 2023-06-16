#%%
import numpy as np
from ultralytics import YOLO
import keras_ocr
import cv2
import cvzone
import math
import imutils as im
import matplotlib.pyplot as plt
from sort import *

pipe = keras_ocr.pipeline.Pipeline()



#%%
# Fungsi untuk mengurutkan 4 edges pada plat agar konsisten
def order_points(pts):

  # Koordinat akan di urutkan dengan urutan kiri-atas
	# kanan-atas, kanan-bawah dan kiri-bawah
  rect = np.zeros((4, 2),dtype="float32")

  # Titik kiri-atas akan memiliki jumlah terkecil
  # dan titik kanan-bawah memiliki jumlah terbesar
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  # Titik kanan-atas akan memiliki hasil pengurangan terkecil
  # dan titik kiri-bawah memiliki hasil pengurangan terbesar
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]

	# return koordinat yang sudah diurutkan
  return rect

# Fungsi untuk trasformasi gambar dengan mengambil gambar didalam area 4 titik
def four_point_transform(images, pts):
	
	(tl, tr, br, bl) = pts
	
  # Menghitung lebar gambar maksimum
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# Menghitung tinggi gambar maksimum
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
  # Transformasi gambar dengan mengambil 4 titik untuk medapatkan 
  # tampak atas dari area gamber yang diingin
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# Menghitung Matriks Transformasi
	M = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective(images, M, (maxWidth, maxHeight))
 
	# Return gambar yang sudah di transformasi
	return warped

# Fungsi untuk menghitung sudut maksimal garis horizontal dari
# Edges plat yang terdeteksi
def count_angle(points) :
  atas_kiri, atas_kanan, bawah_kanan, bawah_kiri = points

  # Menghitung lebar dan tinggi di sepasang sisi
  height_1 = (atas_kanan[1]-atas_kiri[1])
  width_1 = (atas_kanan[0]-atas_kiri[0])
  height_2 = (bawah_kanan[1]-bawah_kiri[1])
  width_2 = (bawah_kanan[0]-bawah_kiri[0])

  # Mencari sudut maksimum relatif terhadap garis horizontal
  # Sudut positif arah jarum jam dari sumbu x positif
  degree_1 = math.atan2(height_1,width_1)
  degree_2 = math.atan2(height_2,width_2)
  if degree_1>=0 and degree_2>=0 :
    degree = max(degree_1,degree_2)
  elif degree_1<0 and degree_2>=0 :
    degree = degree_1
  elif degree_1>=0 and degree_2<0 :
    degree = degree_2
  elif degree_1<0 and degree_2<0 :
    degree = -1*(max(abs(degree_1),abs(degree_2)))
  else :
    degree = 0

  # mengubah dari unit radian ke derajat
  pi = math.pi
  degree_in_degree = (degree/(2*pi)*360)
  
  #return sudut dalam satuan derajat
  return degree_in_degree

# Fungsi untuk lokalisasi kandidat plat kendaraan
# input berupa gambar grayscale yang sudah di inverse (karena plat indonesia
# memiliki karakter cerah dengan latar gelap, algoritma ini awalnya dibuat untuk 
# karakter gelap pada latar cerah
def locate_license_plate_candidates(gray, keep=3):

  # Melakukan blackhat morphological operation untuk menonjolkan area gelap
  # pada latar cerah
  rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
  blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

  # Mencari area terang pada gambar dengan close morphological operation
  squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
  # light = cv2.cvtColor(light, cv2.COLOR_BGR2GRAY)
  light = cv2.threshold(light, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
  # Mempertajam fitur gradient edges dengan metode scharr pada sumbu x
  # Kemudian mengembalikan range nilai piksel ke [0,255]
  gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
  dx=1, dy=0, ksize=-1)
  gradX = np.absolute(gradX)
  (minVal, maxVal) = (np.min(gradX), np.max(gradX))
  gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
  gradX = gradX.astype("uint8")
  
  # Melakukan blur gauss pada gambar kemudian melakukan  closing morphological
  # operation dan kemudian melakukan tresholding dengan metode otsu
  gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
  gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
  # gradX = cv2.cvtColor(gradX, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gradX, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  
  # Melakukan operasi erode dan dilate untuk membuang noise diluar objek signifikan
  thresh = cv2.erode(thresh, None, iterations=3)
  thresh = cv2.dilate(thresh, None, iterations=3)

  # Melakukan masking dengan gambar light hasil fungsi sebelumnya untuk
  # Memfokuskan gambar pada bagian cerah (tempat plat berada)
  # Kemudian melakukan operasi dilate dan erode untuk membuang noise dalam
  # objek signifikan
  thresh = cv2.bitwise_and(thresh, thresh, mask=light)
  thresh = cv2.dilate(thresh, None, iterations=5)
  thresh = cv2.erode(thresh, None, iterations=5)
  
  # Mencari kontur tertutup dari gambar
  cnts,hir = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts=sorted(cnts, key = cv2.contourArea, reverse = True)

  #Template untuk masking
  mask = np.ones(region.shape[:2], dtype="uint8") * 255
  
  # print(len(cnts))
  if len(cnts) >= 4 :
    cv2.drawContours(mask, cnts[4:], -1, 0, -1)
  thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
  # else :
  #   cv2.drawContours(mask, cnts[len(cnts):], -1, 0, -1)

  # Masking untuk kontur yang memiliki ukuran kecil dibanding kontur 
  # objek utama
  for c in cnts :
    if cv2.arcLength(c,True) < 0.25*cv2.arcLength(cnts[0],True) :
      cv2.drawContours(mask, [c], -1, 0, -1)
  thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
  
  cnts,hir = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts=sorted(cnts, key = cv2.contourArea, reverse = True)

  # Membuat boundary box dari semua objek yang diperkirakan sebagai plat 
  # kendaraan pada gambar binary
  points = cv2.findNonZero(thresh)
  xandy, dimension, degree = cv2.minAreaRect(points)
  dimension = list(dimension)
  dimension[0] = 1*dimension[0]+10
  dimension[1] = 1*dimension[1]+10
  rect = []
  rect.append(xandy)
  rect.append(dimension)
  rect.append(degree)
  rect = tuple(rect)

  # Variable untuk menyimpan koordinat 4 titik sudut dari prediksi bounding box
  # untuk plat kendaraan
  box = cv2.boxPoints(rect)
  box = np.int0(box)

  # return kooerdinat dan gambar hasil lokalisasi
  return box,thresh

# Algoritma untuk mencari 4 titik sudut dari lokalisasi plat nomor
def closed_contour_method(images) :
  # Konversi gambar menjadi abu-abu
  gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

  # # Menghilangkan noise dengan iterative bilateral filter
  # d, sigmaColor, sigmaSpace = 5,15,15
  # filtered_img = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

  # Menghilangkan noise dengan GaussianBlur
  filtered_img = cv2.GaussianBlur(gray,(5,5),0)

  # Mencari edges dari dambar dengan algoritma canny
  lower, upper = 50, 150
  edged = cv2.Canny(filtered_img, lower, upper)

  # Mencari kontur tertutup dari gambar kemudian diurutkan dari yang terbesar
  cnts,hir = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
  NumberPlateCnt = np.zeros((4,2))

  # Mencari kontur yang memiliki potensi sebagai objek plat 
  count = 0
  for c in cnts:
        peri = cv2.arcLength(c, True)
        # Aproksimasi panjang kontur yang berpotensi
        epsilon = 0.01 * peri
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        # Memilih kontur dengan 4 titik (quadrilateral)
        if len(approx) == 4  :  
            NumberPlateCnt = approx 
            NumberPlateCnt = NumberPlateCnt.reshape(4,2)
            count += 1
            break

  return NumberPlateCnt




#%%

cap = cv2.VideoCapture("p1.mp4")  # For Video

model = YOLO("best.pt")

classNames = ["plat"]


# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
text=[]
processed_ids = {}
limits = [100, 400, 673, 400]
x1_box, y1_box, x2_box, y2_box = (0, 0,0, 0)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1080,720))
    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # if currentClass == "plat" and conf > 0.3:
            # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
            #                     scale=0.6, thickness=1, offset=3)
            # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(result)
        w, h = x2 - x1, y2 - y1
        region = img[y1:y1 + h, x1:x1 + w]


        if id not in processed_ids:
          cx, cy = x1 + w // 2, y1 + h // 2
          cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)



          if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
              processed_ids[id] = True
              NumberPlateCnt = closed_contour_method(region)

              # Jika cara closed contour tidak dapat mendeteksi
              if NumberPlateCnt.all() == 0 :
                  # Menggunakan metode morphological operation
                  imagen = cv2.bitwise_not(region)
                  imagen = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                  NumberPlateCnt, thresh = locate_license_plate_candidates(imagen)

              # Proses Transformasi gambar
              ordered_points = order_points(NumberPlateCnt)
              warped_images = four_point_transform(region,ordered_points)

              # Menghitung sudut maksimum dari gambar
              degree = count_angle(ordered_points)
              epsilon = 1

              # Error Correction untuk gambar miring dengan sudut kecil
              if abs(degree) < 5 :
                  rotated = im.rotate_bound(warped_images, 2)
              else :
              # Error Correction untuk gambar yang miring
                  if degree <= 0 : # Jika sudut negatif (sisi kiri lebih rendah dari sisi kanan)
                      rotated = im.rotate_bound(warped_images, -1*(0.1*degree) + epsilon)
                  else :  # Jika sudut positif (sisi kanan lebih rendah dari sisi kiri)
                      rotated = im.rotate_bound(warped_images, 0.05*degree + epsilon)
                          
              
              data = []
              data.append(region)
              temp = []
              
              try :
                  predictions = pipe.recognize(data)
                  for imageq, prediction in zip(data, predictions) :
                      for dataq in prediction :
                          temp.append(dataq[0].upper())
              except :
                  temp = []        
          
          
              if temp == [] :
                  # Menggunakan metode morphological operation
                  imagen = cv2.bitwise_not(region)
                  imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                  NumberPlateCnt, thresh = locate_license_plate_candidates(imagen)
                  
                  # Proses Transformasi gambar
                  ordered_points = order_points(NumberPlateCnt)
                  warped_images = four_point_transform(region,ordered_points)

                  # Menghitung sudut maksimum dari gambar
                  degree = count_angle(ordered_points)
                  epsilon = 1

                  # Error Correction untuk gambar miring dengan sudut kecil
                  if abs(degree) < 5 :
                      rotated = im.rotate_bound(warped_images, 2)
                  else :
                  # Error Correction untuk gambar yang miring
                      if degree <= 0 : # Jika sudut negatif (sisi kiri lebih rendah dari sisi kanan)
                          rotated = im.rotate_bound(warped_images, -1*(0.1*degree) + epsilon)
                      else :  # Jika sudut positif (sisi kanan lebih rendah dari sisi kiri)
                          rotated = im.rotate_bound(warped_images, 0.05*degree + epsilon)

                  # Menampilkan data hasil pembacaan Keras-OCR
                  data = []
                  data.append(rotated)
                  temp = []

                  try :
                      predictions = pipe.recognize(data)
                      for imageq, prediction in zip(data, predictions) :
                          for dataq in prediction :
                              temp.append(dataq[0].upper())
                  except :
                      temp = []

                      
              if temp==[] or len(temp)<3:
                  
                  pass
              else :
                  temp = temp[:3]
                  i=0
                  while temp[0].isnumeric() or temp[1].isalpha() or temp[2].isnumeric or i==15:
                      i += 1
                      rotated=im.rotate_bound(rotated,i)

                      images =[rotated]
                      temp=[]
                      ocr_result = pipe.recognize(images)
                      for rotated, ocr_result in zip(images,ocr_result):
                          for images in ocr_result:
                              temp.append(images[0].upper())
                      if len(temp)<3:
                          break
                      else:
                          temp=temp[:3]
                          
                          break
              text = str(temp)
              cvzone.putTextRect(img, f' {text}', (max(0, x1), max(35, y1)),
                                 scale=2, thickness=3, offset=0)
              




        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
    
        
    cv2.putText(img,str(text),(10,100),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),4)
    


    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    p = cv2.waitKey(1)
    if p == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
# %%
