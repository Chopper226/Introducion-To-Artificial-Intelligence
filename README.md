# 人工智慧導論
此為 111 學年度 **人工智慧導論 I & II** 課程學習之紀錄  
內含 Teachable Machine 影像辨識 、 Face Recognition 人臉辨識 、 監督式學習 ( kNN 、 Decision Tree ) 、非監督式學習 ( kMeans )

## Teachable Machine 影像辨識

### 實作方式
1. 利用 [Teachable Machine](https://teachablemachine.withgoogle.com/)  進行影像辨識，蒐集 3-4 組不同類別的照片，每組各 40 張
2. 調整 Epochs 以及 Batch size 的值 ( Learning Rate 維持 0.001 ) 並開始訓練模型
3. 系統將隨機選取 85\%的圖片數當作訓練資料，15\%當作測試資料
4. 訓練結束後，觀察模型訓練過程的正確率與 Loss 數值資訊
5. 計算每類別的測試正確率與混淆矩陣，找出最適合的模型

## Face Recognition 人臉辨識

### 實作方式
1. 使用 Python 的 OpenCV 函式庫以及 [Harr 特徵分類器人臉模型](https://github.com/atduskgreg/opencv-processing/tree/master/lib/cascade-files) 進行人臉辨識
2. 建立CascadeClassifier 物件
3. 更改 scaleFactor、minNeighbors、minSize/maxSize 的值找尋照片中的臉孔
4. 將每個臉孔進行圖形擷取 ( 用長方形框起來 )
5. 利用偵測到的人臉中範圍，再進一步偵測眼睛 / 笑容
6. 更改 scaleFactor、minNeighbors、minSize/maxSize 的值找尋長方形中的想偵測的特徵
7. 將每個特徵進行圖形擷取 ( 用長方形框起來 )

## 監督式學習 ( kNN 、 Decision Tree )

### 實作方式

## 非監督式學習 ( kMeans )

### 訓練與驗證方式
1. 將分群結果投射於座標系統
2. 訓練模型 - 更改 K 值
3. 從分好的群中隨機取三群
4. 將這三群的前十筆資料印出
5. 將這三群的前十筆資料化成圖像觀察分類結果
