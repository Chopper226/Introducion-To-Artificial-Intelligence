import cv2
img1 = cv2.imread("Monster.jpg",1) # 匯入照片

''' 臉部 '''
case_path1 = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(case_path1)

''' 眼睛 '''
case_path2 = cv2.data.haarcascades + "haarcascade_eye.xml"
eyesCascade = cv2.CascadeClassifier(case_path2)

''' 臉部辨識 '''
faces = faceCascade.detectMultiScale( img1 , scaleFactor=1.2, minNeighbors=3 , minSize=(10,10) , flags=cv2.CASCADE_SCALE_IMAGE )  
# print("偵測" + str(len(faces)) +"張人臉")

''' 眼睛辨識 '''
for (x, y, w, h) in faces:
  img2 = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2) # 人臉的長方形
  face_eye = img2[y:y+h, x:x+w]
  eyes = eyesCascade.detectMultiScale( face_eye , scaleFactor=1.1 , minNeighbors=50 , minSize=(25, 25) , flags=cv2.CASCADE_SCALE_IMAGE) 
  for (x,y,w,h) in eyes:
       cv2.rectangle(face_eye, (x,y), (x+w, y+h), (255,0,0) ,2) # 眼睛的長方形

cv2.imshow( "Image",img2) # 顯示圖片
cv2.waitKey(0)
