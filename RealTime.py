import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(3,640)# 320 320 640 720
capture.set(4,480)#180 240  360 405

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv@2/2.4.13.7_3/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
smile_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_smile.xml')

while True:
    ret, img = capture.read()
    img = cv2.flip(img,1)#鏡表示にするため．
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
    if len(faces) == 1 :
        for (x,y,w,h) in faces:
            # 検知した顔を矩形で囲む
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # 顔画像（グレースケール）
            roi_gray = gray[y:y+h, x:x+w]
            # 顔画像（カラースケール）
            roi_color = img[y:y+h, x:x+w]
            # 顔画像（グレースケール/上半分）
            eye_gray = gray[y:y+h*3//4, x:x+w]
            # 顔画像（カラースケール/上半分）
            eye_color = img[y:y+h*3//4, x:x+w]
            # cv2.imshow("eye_gray",eye_gray) #確認のためサイズ統一させた画像を表示
            # 顔の中から目を検知(緑)
            eyes = eye_cascade.detectMultiScale(eye_gray,scaleFactor=1.1,minNeighbors=5)
            if len(eyes) == 2 :
                for (ex,ey,ew,eh) in eyes:
                    # 検知した目を矩形で囲む
                    cv2.rectangle(eye_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            # # 顔の中から口を検知(赤)
            # for k in range(1,50) :
            #     for j in range(1,20) :
            #         mouth = mouth_cascade.detectMultiScale(roi_gray,scaleFactor=1.1,minNeighbors=k,minSize=(j,j))
            #         if len(mouth) == 1 :
            #             print("3. Mouth")
            #             print("minNeighbors  " , k)
            #             print("     minSize (" , j , "," , j , ")")
            #             for (mx,my,mw,mh) in mouth:
            #                 # 検知した目を矩形で囲む
            #                 cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
            #             break
            #     else:
            #         continue
            #     break

            # 顔画像（グレースケール/下半分）
            mouth_gray = gray[y+h*1//2:y+h, x:x+w]
            # 顔画像（カラースケール/下半分）
            mouth_color = img[y+h*1//2:y+h, x:x+w]
            cv2.imshow("mouth_gray",mouth_gray) #確認のためサイズ統一させた画像を表示
            # 笑顔検出(赤)
            smile = smile_cascade.detectMultiScale(mouth_gray,scaleFactor=1.1,minNeighbors=5)
            if len(smile) == 1 :
                for(sx,sy,sw,sh) in smile:
                    cv2.rectangle(mouth_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)

            # 笑顔度
            # サイズを拡大
            smile_gray = cv2.resize(roi_gray,(100,100))
            # cv2.imshow("smile_gray",mouth_gray) #確認のためサイズ統一させた画像を表示
            # 輝度で規格化
            lmin = smile_gray.min() #輝度の最小値
            lmax = smile_gray.max() #輝度の最大値
            for index1, item1 in enumerate(smile_gray):
                for index2, item2 in enumerate(item1) :
                    smile_gray[index1][index2] = int((item2 - lmin)/(lmax-lmin) * item2)
            # cv2.imshow("roi_gray2",roi_gray)  #確認のため輝度を正規化した画像を表示
            smile = smile_cascade.detectMultiScale(smile_gray,scaleFactor=1.1,minNeighbors=0,minSize=(20, 20))
            if len(smile) > 0 :
                # サイズを考慮した笑顔認識
                smile_neighbors = len(smile)
                # print("smile_neighbors=",smile_neighbors) #確認のため認識した近傍矩形数を出力
                LV = 2/100
                intensityZeroOne = smile_neighbors * LV
                if intensityZeroOne > 1.0 :
                    intensityZeroOne = 1.0
                # print(intensityZeroOne) #確認のため強度を出力
                # for(sx,sy,sw,sh) in smile:
                #     cv2.rectangle(mouth_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)

    height, width, channel = img.shape # height,widthの順
    # print(width,height)
    # 画像に文字列を追加
    # 右下
    # cv2.putText(img, str(int(intensityZeroOne*100)) + "%", (width-70, height-10) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 4)
    # 左上
    if int(intensityZeroOne*100) > 90 :
        cv2.putText(img, str(int(intensityZeroOne*100)) + "%", (10, 35) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)
    else :
        cv2.putText(img, str(int(intensityZeroOne*100)) + "%", (10, 35) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 4)
    cv2.imshow('img',img)
    # key Operation
    key = cv2.waitKey(5)
    if key ==27 or key ==ord('q'): #escまたはeキーで終了
        break

capture.release()
cv2.destroyAllWindows()
print("Exit")
