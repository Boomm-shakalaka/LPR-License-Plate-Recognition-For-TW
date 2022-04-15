import numpy as np
import cv2
import sys
import os
class Carplate_detection:
    def __init__(self,car):
        self.car=car
        self.plate=None
        self.plate_thre=None
        self.letter_image_regions=None
        self.letterlist=None
        self.nChar=None
        self.nn=None
        self.bg=None
        self.lifearea=None
        self.real_shape=None

    def app_path(self):
        """Returns the base application path."""
        if hasattr(sys, 'frozen'):
            # Handles PyInstaller
            return os.path.dirname(sys.executable)  # 使用pyinstaller打包后的exe目录
        return os.path.dirname(__file__)  # 没打包前的py目录

    def haar_detect(self):
        global plate
        self.car = cv2.resize(self.car, (600, 425))  # 尺寸300x225
        file = self.app_path() + r'/haar_model/haar_carplate.xml'
        self.is_file = os.path.exists(file)
        if self.is_file== False:
            return
        detector = cv2.CascadeClassifier(file)
        signs = detector.detectMultiScale(self.car, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))  # 框出車牌
        x,y,w,h=0,0,0,0
        if len(signs) > 0:
            for (x, y, w, h) in signs:
                self.plate = self.car[y:y + h, x:x + w]
                self.plate= cv2.resize(self.plate, (140, 40))
            self.car=cv2.rectangle(self.car, (x,y), (x+w,y+h), (0,255,0), 1)

    def plate_findContours(self):
        img_gray = cv2.cvtColor(self.plate, cv2.COLOR_BGR2GRAY)  # 灰階
        _, img_thre = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)  # 黑白,黑底白字
        self.plate_thre=img_thre
        contours1 = cv2.findContours(self.plate_thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 尋找輪廓
        contours = contours1[0]  # 取得輪廓
        self.letter_image_regions = []  # 文字圖形串列
        for contour in contours:  # 依序處理輪廓
            (x, y, w, h) = cv2.boundingRect(contour)  # 單一輪廓資料
            self.letter_image_regions.append((x, y, w, h))  # 輪廓資料加入串列
        self.letter_image_regions = sorted(self.letter_image_regions, key=lambda x1: x1[0])  # 按X坐標排序

        return self.letter_image_regions
        #return img_thre  # ,findcontours只能将白色作为要识别的物体


    def plate_findLetter(self):
        count = 0
        for box in self.letter_image_regions:  # 依序處理輪廓資料
            x, y, w, h = box
            # x 必須介於 2~125 且寬度在 5~26、高度在 20~39 才是文字
            if x >= 2 and x <= 125 and w >= 5 and w <= 26 and h >= 20 and h < 40:
                count += 1
        if count < 6:  # 若字元數不足，可能是有兩個字元連在一起，將字元寬度放寬再重新擷取
            wmax = 35
        else:
            wmax = 26  # 正常字元寬度
        self.nChar = 0  # 計算共擷取多少個字元
        self.letterlist = []  # 儲存擷取的字元坐標
        for box in self.letter_image_regions:  # 依序處理輪廓資料
            x, y, w, h = box
            # x 必須介於 2~125 且寬度在 5~wmax、高度在 20~39 才是文字
            if x >= 2 and x <= 125 and w >= 5 and w <= wmax and h >= 20 and h < 40:
                self.nChar += 1
                self.letterlist.append((x, y, w, h))  # 儲存擷取的字元
        #return letterlist, nChar

    def plate_Clean(self):
        for i in range(len(self.plate_thre)):
            for j in range(len(self.plate_thre[i])):
                if self.plate_thre[i][j] == 255:
                    count = 0
                    for k in range(-2, 3):
                        for l in range(-2, 3):
                            try:
                                if self.plate_thre[i + k][j + l] == 255:
                                    count += 1
                            except IndexError:
                                pass
                    if count <= 6:
                        self.plate_thre[i][j] = 0

        self.real_shape = []

        def area_Search(row, col):
            if self.bg[row][col] != 255:
                return
            self.bg[row][col] = self.lifearea  # 記錄生命區的編號
            if col > 1:  # 左方
                if self.bg[row][col - 1] == 255:
                    self.nn += 1
                    area_Search(row, col - 1)
            if col < w - 1:  # 右方
                if self.bg[row][col + 1] == 255:
                    self.nn += 1
                    area_Search( row, col + 1)
            if row > 1:  # 上方
                if self.bg[row - 1][col] == 255:
                    self.nn += 1
                    area_Search(row - 1, col)
            if row < h - 1:  # 下方
                if self.bg[row + 1][col] == 255:
                    self.nn += 1
                    area_Search(row + 1, col)

        for i, box in enumerate(self.letterlist):  # 依序擷取所有的字元
            x, y, w, h = box
            self.bg = self.plate_thre[y:y + h, x:x + w]
            # 去除崎鄰地
            if i == 0 or i == self.nChar:  # 只去除第一字元和最後字元的崎鄰地
                self.lifearea = 0  # 生命區塊
                self.nn = 0  # 每個生命區塊的生命數
                life = []  # 記錄每個生命區塊的生命數串列
                for row in range(0, h):
                    for col in range(0, w):
                        if self.bg[row][col] == 255:
                            self.nn = 1  # 生命起源
                            self.lifearea = self.lifearea + 1  # 生命區塊數
                            area_Search(row, col)  # 以生命起源為起點探索每個生命區塊的總生命數
                            life.append(self.nn)
                maxlife = max(life)  # 找到最大的生命數
                indexmaxlife = life.index(maxlife)  # 找到最大的生命數的區塊編號
                for row in range(0, h):
                    for col in range(0, w):
                        if self.bg[row][col] == indexmaxlife + 1:
                            self.bg[row][col] = 255
                        else:
                            self.bg[row][col] = 0
            self.real_shape.append(self.bg)  # 加入字元
        #return self.real_shape

    def new_Plate(self):
        # 在圖片週圍加白色空白OCR才能辨識
        newH, newW = self.plate_thre.shape
        space = 8  # 空白寬度
        offset = 2
        self.New_plate = np.zeros((newH + space * 2, newW + space * 2 + self.nChar * 3, 1), np.uint8)  # 建立背景
        self.New_plate.fill(0)  # 背景黑色

        # 將車牌文字加入黑色背景圖片中
        for i, letter in enumerate(self.real_shape):
            h = letter.shape[0]  # 原來文字圖形的高、寬
            w = letter.shape[1]
            x = self.letterlist[i][0]  # 原來文字圖形的位置
            y = self.letterlist[i][1]

            for row in range(h):  # 將文字圖片加入背景
                for col in range(w):
                    self.New_plate[space + y + row][space + x + col + i * offset] = letter[row][col]  # 擷取圖形
        _, self.New_plate = cv2.threshold(self.New_plate, 127, 255, cv2.THRESH_BINARY_INV)  # 轉為白色背景、黑色文字
        self.New_plate = cv2.resize(self.New_plate,(150,70))
        return self.New_plate

