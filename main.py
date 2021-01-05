from google.colab.patches import cv2_imshow
import sys
import os
import dlib
import glob
import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import imutils
import warnings
warnings.filterwarnings('ignore')

def show_img(path):

  img = cv2.imread(path)
  img = cv2.resize(img, (128,128),interpolation=cv2.INTER_CUBIC)
  cv2_imshow(img)

# 人臉68特徵點模型路徑
predictor_path = './shape_predictor_68_face_landmarks.dat'

# 人臉辨識模型路徑
face_rec_model_path = './dlib_face_recognition_resnet_model_v1.dat'

# 比對人臉圖片資料夾名稱
faces_folder_path = './train'

# 需要辨識的人臉圖片名稱
img_path = './trash.jpg'

# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()

# 載入人臉特徵點檢測器
sp =  dlib.shape_predictor(predictor_path)

# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 比對人臉描述子列表
descriptors = []

# 比對人臉名稱列表

candidate = []
# 針對比對資料夾裡每張圖片做比對:
# 1.人臉偵測
# 2.特徵點偵測
# 3.取得描述子

for f in glob.glob(os.path.join(faces_folder_path, '*.jpg')):
    base = os.path.basename(f)
    # 依序取得圖片檔案人名
    candidate.append(os.path.splitext(base)[0])
    img = io.imread(f)
    # 1.人臉偵測
    dets = detector(img, 1)

    for (k, d) in enumerate(dets):
        # 2.特徵點偵測
        shape = sp(img, d)
        # 3.取得描述子，128維特徵向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        # 轉換numpy array格式
        v = np.array(face_descriptor)
        descriptors.append(v)
# 針對需要辨識的人臉同樣進行處理

img = io.imread(img_path)

dets = detector(img, 1)
dist = []
pil_image = Image.fromarray(img)
draw = ImageDraw.Draw(pil_image)
for (k, d) in enumerate(dets):
    dist = []
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test = np.array(face_descriptor)

    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()

  # 計算歐式距離

    for i in descriptors:
        dist_ = np.linalg.norm(i - d_test)
        dist.append(dist_)

  # 將比對人名和比對出來的歐式距離組成一個dict

    c_d = dict(zip(candidate, dist))
    print(c_d)

  # 根據歐式距離由小到大排序

    cd_sorted = sorted(c_d.items(), key=lambda d: d[1])

  # 取得最短距離就為辨識出的人名

    rec_name = cd_sorted[0][0]
    print(cd_sorted[0][0],cd_sorted[0][1])
    show_img(img_path)
    show_img(f'{faces_folder_path}/{cd_sorted[0][0]}.jpg')






