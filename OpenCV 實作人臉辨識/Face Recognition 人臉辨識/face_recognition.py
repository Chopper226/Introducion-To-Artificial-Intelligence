import cv2
img1 = cv2.imread("MayDay.jpg",1) # 匯入照片

''' 臉部 '''
case_path1 = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(case_path1)


''' 臉部辨識 '''
faces = faceCascade.detectMultiScale( img1 , scaleFactor=1.1, minNeighbors=5 , minSize=(10,10) , flags=cv2.CASCADE_SCALE_IMAGE )  
# print("偵測" + str(len(faces)) +"張人臉")

for (x,y,w,h) in faces:
    cv2.rectangle(img1, (x,y), (x+w, y+h), (0,0,255) ,2)

cv2.imshow( "Image",img1) # 顯示圖片
cv2.waitKey(0)
# 圖片來源 : https://stars.udn.com/star/story/10092/6072651
