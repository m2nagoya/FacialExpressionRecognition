import cv2
import numpy

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv@2/2.4.13.7_3/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
smile_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_smile.xml')

# イメージファイルの読み込み
img = cv2.imread('TEST_IMAGE/001_IMG.png')
# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔を検知

for j in range(100,200) :
    for i in range(50,5,-1) :
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=i,minSize=(j,j))
        if len(faces) == 8 :
            for (x,y,w,h) in faces:
                print("1. Face")
                print("minNeighbors  " , i)
                print("     minSize  " , j)
                # 検知した顔を矩形で囲む
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                # 顔画像（グレースケール）
                roi_gray = gray[y:y+h, x:x+w]
                # 顔画像（カラースケール）
                roi_color = img[y:y+h, x:x+w]

                # 顔画像（グレースケール/上半分）
                eye_gray = gray[y:y+h*2//3, x:x+w]
                # 顔画像（カラースケール/上半分）
                eye_color = img[y:y+h*2//3, x:x+w]
                # cv2.imshow("eye_gray",eye_gray) #確認のためサイズ統一させた画像を表示
                # 顔の中から目を検知(緑)
                for k in range(50,5,-1) :
                    eyes = eye_cascade.detectMultiScale(eye_gray,scaleFactor=1.1,minNeighbors=k)
                    if len(eyes) == 2 :
                        print("2. Eye")
                        print("minNeighbors  " , k)
                        for (ex,ey,ew,eh) in eyes:
                            # 検知した目を矩形で囲む
                            cv2.rectangle(eye_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                        break

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
                mouth_gray = gray[y+h*1//3:y+h, x:x+w]
                # 顔画像（カラースケール/下半分）
                mouth_color = img[y+h*1//3:y+h, x:x+w]
                # cv2.imshow("mouth_gray",mouth_gray) #確認のためサイズ統一させた画像を表示
                # 笑顔検出(赤)
                for m in range(50,5,-1) :
                    smile = smile_cascade.detectMultiScale(mouth_gray,scaleFactor=1.1,minNeighbors=m)
                    if len(smile) == 1 :
                        print("3. Mouth")
                        print("minNeighbors  " , m)
                        for(sx,sy,sw,sh) in smile:
                            cv2.rectangle(mouth_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
                        break

                # 笑顔度
                # サイズを縮小
                smile_gray = cv2.resize(roi_gray,(w,h))
                # cv2.imshow("smile_gray",mouth_gray) #確認のためサイズ統一させた画像を表示
                # 輝度で規格化
                lmin = smile_gray.min() #輝度の最小値
                lmax = smile_gray.max() #輝度の最大値
                for index1, item1 in enumerate(smile_gray):
                    for index2, item2 in enumerate(item1) :
                        smile_gray[index1][index2] = int((item2 - lmin)/(lmax-lmin) * item2)
                # cv2.imshow("roi_gray2",roi_gray)  #確認のため輝度を正規化した画像を表示
                smile = smile_cascade.detectMultiScale(smile_gray,scaleFactor=1.1,minNeighbors=0,minSize=(j//5,j//5))
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

                    # print("smile_neighbors=",smile_neighbors) #確認のため認識した近傍矩形数を出力
                    # print(intensityZeroOne) #確認のため強度を出力
                    # 画像を保存
                    # cv2.imwrite("Result.png",img)
                    height, width, channel = img.shape # height,widthの順
                    # print(width,height)
                    # 画像に文字列を追加
                    # 右下
                    cv2.putText(img, str(int(intensityZeroOne*100)) + "%", (x+w-30, y+h+30) , cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 4)
                    # 左上
                    # cv2.putText(img, str(int(intensityZeroOne*100)) + "%", (10, 35) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 4)
                else :
                    height, width, channel = img.shape # height,widthの順
                    # print(width,height)
                    # 画像に文字列を追加
                    # 右下
                    cv2.putText(img, "0%", (x+w-30, y+h+30) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)
                    # 左上
                    # cv2.putText(img, str(int(intensityZeroOne*100)) + "%", (10, 35) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 4)
            break
    else :
        continue
    break

cv2.imwrite("Result.png", img)
# 画像表示
cv2.imshow('img',img)

print("\nFinish")
# 何かキーを押したら終了
cv2.waitKey(0)
cv2.destroyAllWindows()


# 参考文献
# OpenCVを使った顔認識（Haar-like特徴分類器）
# [ https://qiita.com/hitomatagi/items/04b1b26c1bc2e8081427 ]
# 目を検出する ついでに口・鼻も
# [ http://nobotta.dazoo.ne.jp/blog/?p=503 ]
# 「顔以外」のものを画像認識する on iOS
# [ https://qiita.com/shu223/items/ffd2202eaf92d342f83d ]
# PythonとOpenCVを使った笑顔認識
# [ https://qiita.com/fujino-fpu/items/99ce52950f4554fbc17d ]
