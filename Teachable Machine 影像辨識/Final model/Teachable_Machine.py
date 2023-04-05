from numpy.lib.npyio import loadtxt
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

model = load_model('keras_model.h5', compile=False)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open('C++.png').convert('RGB') #測試檔案
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

prediction = model.predict(data)
index = np.argmax(prediction)

labels = open('labels.txt', 'r').readlines()
class_name = labels[index][2:6]

confidence_score = prediction[0][index]

print("Class: ", class_name)
print("Confidence Score: ", confidence_score)