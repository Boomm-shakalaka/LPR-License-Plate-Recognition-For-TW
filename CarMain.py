import shutil
import time
import easyocr
import threading
import pyocr
import numpy as np
import os
from PyQt5 import QtCore
from PyQt5.Qt import *
import sys
import cv2
from PIL import Image
import HAAR_detect
import CTPN_detect
import YOLO_detect
class Yolo_Thread(QThread):
    _signal = pyqtSignal(np.ndarray, np.ndarray, bool, bool)
    def __init__(self, car):
        super().__init__()
        self.car = car
        self.new_plate = None
        self.isfile_flag = True
        self.isplate_flag = True
        self.working = True

    def __del__(self):
        self.wait()

    def run(self):
        Capacity = YOLO_detect.dark_detection(self.car)
        Capacity.darknet_detection()
        if Capacity.weights_isfile == False or Capacity.classes_isfile== False or Capacity.cfg_isfile== False:
            self.isfile_flag = False
            self._signal.emit(np.array([1]), np.array([1]), self.isfile_flag, self.isplate_flag)
        elif Capacity.plate is None:
            self.isplate_flag = False
            self._signal.emit(np.array([1]), np.array([1]), self.isfile_flag, self.isplate_flag)
        else:
            self.new_plate = Capacity.plate

            self._signal.emit(Capacity.car, self.new_plate, self.isfile_flag, self.isplate_flag)

class Haar_Thread(QThread):
    _signal =pyqtSignal(np.ndarray,np.ndarray,bool,bool)
    def __init__(self,car):
        super().__init__()
        self.car=car
        self.new_plate =None
        self.isfile_flag = True
        self.isplate_flag= True
        self.working = True

    def __del__(self):
        self.wait()

    def run(self):
        Capacity = HAAR_detect.Carplate_detection(self.car)
        Capacity.haar_detect()
        if Capacity.is_file == False:
            self.isfile_flag=False
            self._signal.emit(np.array([1]), np.array([1]), self.isfile_flag,self.isplate_flag)
        elif Capacity.plate is None:  # 判断是否侦测到车牌
            self.isplate_flag= False
            self._signal.emit(np.array([1]),np.array([1]),self.isfile_flag,self.isplate_flag)
        else:
            Capacity.plate_findContours()  # 储存文字参数
            Capacity.plate_findLetter()
            Capacity.plate_Clean()
            self.new_plate = Capacity.new_Plate()
            self._signal.emit(Capacity.car,self.new_plate,self.isfile_flag,self.isplate_flag)

class Ctpn_Thread(QThread):
    _signal =pyqtSignal(np.ndarray,np.ndarray,bool,bool)
    def __init__(self,car):
        super().__init__()
        self.car=car
        self.new_plate =None
        self.isfile_flag = True
        self.isplate_flag= True
        self.working = True

    def __del__(self):
        self.wait()

    def run(self):
        Capacity = CTPN_detect.detection(self.car)
        Capacity.ctpn_detect()
        if Capacity.is_file == False:
            self.isfile_flag = False
            self._signal.emit(np.array([1]), np.array([1]), self.isfile_flag, self.isplate_flag)
        elif Capacity.plate is None:
            self.isplate_flag = False
            self._signal.emit(np.array([1]), np.array([1]), self.isfile_flag, self.isplate_flag)
        else:
            self.new_plate = Capacity.plate
            self._signal.emit(Capacity.car,self.new_plate,self.isfile_flag,self.isplate_flag)

class win(QWidget):
    def __init__(self, parent=None):
        super(win,self).__init__(parent)
        self.car = np.array([1])
        self.real_car=None
        self.file_Name=None
        self.file_Path=None
        self.txt=""
        self.ocr_flag=False
        self.rbt_Modeltype="haar"
        self.rbt_Ocrtype="pyocr"
        self.setGeometry(100, 100, 1280, 720)
        self.setWindowTitle('Plate')

        ##图片显示##
        self.ori_label = QLabel('ori_label')
        self.ori_label.setFrameShape(QFrame.Box)
        self.ori_label.setFrameShadow(QFrame.Raised)# 设置阴影 只有加了这步才能设置边框颜色
        self.ori_label.setLineWidth(5)
        self.ori_label.setStyleSheet('background-color: rgb(125, 125, 125)') # 设置背景颜色，包括边框颜色
        self.ori_label.setPixmap(QPixmap('pic/bg_gray.jpg').scaled(QSize(600,425)))  # 设置显示的图片
        self.ori_label.setFixedSize(600,425)#窗口大小

        ##扫面后的图片##
        self.scan_label = QLabel('scan_label')
        self.scan_label.setFrameShape(QFrame.Box)
        self.scan_label.setFrameShadow(QFrame.Raised)
        self.scan_label.setLineWidth(1)
        self.scan_label.setStyleSheet('background-color: rgb(0, 0, 0)')
        self.scan_label.setPixmap(QPixmap('pic/bg_gray.jpg').scaled(QSize(250, 120)))
        self.scan_label.setAlignment(Qt.AlignCenter)
        self.scan_label.setFixedSize(300,150)#窗口大小

        '''定义創建Tree部件'''
        self.model01 = QFileSystemModel()
        self.model01.setRootPath(os.getcwd())
        self.model01.setFilter(QtCore.QDir.Dirs | QtCore.QDir.NoDotAndDotDot)# 进行筛选只显示文件夹，不显示文件和特色文件
        self.model01.setRootPath(QDir.currentPath())
        self.treeView1 = QTreeView(self)
        self.treeView1.setAnimated(False)
        self.treeView1.setObjectName('treeView')
        self.treeView1.setFixedSize(300, 400)
        self.treeView1.setModel(self.model01)
        self.treeView1.setColumnWidth(0, 200)  # 欄位寬度，一定要放在setModel()後面
        for col in range(1, 4):
            self.treeView1.setColumnHidden(col, True)

        #定义创建右边窗口
        self.model02 = QStandardItemModel()
        self.treeView2 = QTreeView(self)
        self.treeView2.setFixedSize(300,400)#窗口大小
        self.treeView2.setModel(self.model02)

        '''定義提示框部件'''
        self.textEdit = QTextEdit()
        self.textEdit.setFixedSize(600, 200)  # 窗口大小
        self.textEdit.setObjectName('tipEdit')
        self.textEdit.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.textEdit.setText('Tip:')
        self.textEdit.setReadOnly(True)  # 设置为只读，即可以在代码中向textEdit里面输入，但不能从界面上输入,没有这行代码即可以从界面输入

        ##输出文字##
        self.txt_label = QLabel('plate_Number')
        self.txt_label.setFrameShape(QFrame.Box)#边框样式
        self.txt_label.setFrameShadow(QFrame.Plain)#边框样式
        self.txt_label.setLineWidth(1)
        self.txt_label.setStyleSheet('background-color: rgb(255, 255, 255)')
        self.txt_label.setFixedSize(300,150)  # 窗口大小
        self.txt_label.setFont(QFont("Roman times",20,QFont.Bold))
        self.txt_label.setAlignment(Qt.AlignCenter)

        ##按钮##
        self.open_bt = QPushButton('OpenFile', self)
        self.start_bt = QPushButton('Start', self)
        self.save_bt = QPushButton('Save', self)
        self.open_bt.setFixedSize(120, 60)
        self.start_bt.setFixedSize(120, 60)
        self.save_bt.setFixedSize(120, 60)

        ##Radio部件##
        self.model_haar = QRadioButton('Haar')
        self.model_haar.setObjectName('radio_button')
        self.model_ctpn = QRadioButton('CTPN')
        self.model_ctpn.setObjectName('radio_button')
        self.model_yolo = QRadioButton('Yolo')
        self.model_yolo.setObjectName('radio_button')
        self.py_ocr = QRadioButton('Pyocr  ')
        self.py_ocr.setObjectName('radio_button')
        self.easy_ocr=QRadioButton('Easyocr')
        self.easy_ocr.setObjectName('radio_button')
        self.btngroup1 = QButtonGroup()
        self.btngroup1.addButton(self.model_haar, 1)
        self.btngroup1.addButton(self.model_ctpn, 2)
        self.btngroup1.addButton(self.model_yolo, 3)
        self.btngroup2 = QButtonGroup()
        self.btngroup2.addButton(self.py_ocr, 1)
        self.btngroup2.addButton(self.easy_ocr, 2)

        ##左框架##
        self.left_box = QVBoxLayout()
        self.left_box.addWidget(self.ori_label, 0, Qt.AlignLeft)
        self.left_box.addWidget(self.textEdit)

        ##文本框架##
        self.label_box=QHBoxLayout()
        self.label_box.addWidget(self.scan_label)
        self.label_box.addWidget(self.txt_label)

        ##Radio Button框架##
        self.radio_box =  QVBoxLayout()
        self.model_box = QHBoxLayout()
        self.model_box.addWidget(self.model_haar)
        self.model_box.addWidget(self.model_ctpn)
        self.model_box.addWidget(self.model_yolo)
        self.ocr_box = QHBoxLayout()
        self.ocr_box.addWidget(self.py_ocr)
        self.ocr_box.addWidget(self.easy_ocr)
        self.radio_box.addLayout(self.model_box)
        self.radio_box.addLayout(self.ocr_box)

        ##按钮框架##
        self.bt_box = QHBoxLayout()
        self.bt_box.addStretch(1)
        self.bt_box.addLayout(self.radio_box)
        self.bt_box.addStretch(1)
        self.bt_box.addWidget(self.open_bt)
        self.bt_box.addWidget(self.start_bt)
        self.bt_box.addWidget(self.save_bt)


        self.bt_box.addStretch(1)

        ##目录框架##
        self.tree_box= QHBoxLayout()
        self.tree_box.addWidget(self.treeView1)
        self.tree_box.addWidget(self.treeView2)

        ##右框架##
        self.right_box = QVBoxLayout()
        self.right_box.addLayout(self.label_box)
        self.right_box.addLayout(self.bt_box)
        self.right_box.addLayout(self.tree_box)
        self.vbox = QHBoxLayout()
        self.vbox.addLayout(self.left_box)
        self.vbox.addLayout(self.right_box)
        self.setLayout(self.vbox)

        ##初始##
        self.start_bt.setEnabled(False)
        self.save_bt.setEnabled(False)
        self.model_haar.setChecked(True)
        self.py_ocr.setChecked(True)


        ## 动作##
        self.treeView1.doubleClicked.connect(self.get_Treepath)
        self.treeView2.clicked.connect(self.path_Clicked)
        self.treeView2.doubleClicked.connect(self.path_DoubleClicked)
        self.open_bt.clicked.connect(self.open_Pic)
        self.start_bt.clicked.connect(self.start_Rec)
        self.save_bt.clicked.connect(self.saveSlot)
        self.btngroup1.buttonClicked.connect(self.rbt_model_Clicked)
        self.btngroup2.buttonClicked.connect(self.rbt_ocr_Clicked)

    def rbt_model_Clicked(self):
        sender = self.sender()
        if sender == self.btngroup1:
            if self.btngroup1.checkedId() == 1:
                self.textEdit.append('Model is haar')
                self.textEdit.append('---------------')
                self.textEdit.moveCursor(QTextCursor.End)
                self.rbt_Modeltype="haar"
            elif self.btngroup1.checkedId() == 2:
                self.textEdit.append('Model is ctpn')
                self.textEdit.append('---------------')
                self.textEdit.moveCursor(QTextCursor.End)
                self.rbt_Modeltype = "ctpn"
            elif self.btngroup1.checkedId() == 3:
                self.textEdit.append('Model is Yolo')
                self.textEdit.append('---------------')
                self.textEdit.moveCursor(QTextCursor.End)
                self.rbt_Modeltype = "yolo"

    def rbt_ocr_Clicked(self):
        sender = self.sender()
        if sender == self.btngroup2:
            if self.btngroup2.checkedId() == 1:
                self.textEdit.append('Ocr is pyocr')
                self.textEdit.append('---------------')
                self.textEdit.moveCursor(QTextCursor.End)
                self.rbt_Ocrtype = "pyocr"
            elif self.btngroup2.checkedId() == 2:
                self.textEdit.append('Ocr is easyocr')
                self.textEdit.append('---------------')
                self.textEdit.moveCursor(QTextCursor.End)
                self.rbt_Ocrtype = "easyocr"

    def get_Treepath(self, Qmodelidx):
        self.model02.clear() # 每次点击清空右边窗口数据
        PathData = []# 定义一个数组存储路径下的所有文件
        filePath = self.model01.filePath(Qmodelidx)# 获取双击后的指定路径
        PathDataName = self.model02.invisibleRootItem() # List窗口文件赋值
        PathDataSet = os.listdir(filePath)        # 拿到文件夹下的所有文件
        PathDataSet.sort()# 进行将拿到的数据进行排序
        for Data in range(len(PathDataSet)):       # 遍历判断拿到的文件是文件夹还是文件，Flase为文件，True为文件夹
            if os.path.isdir(filePath + '\\' + PathDataSet[Data]) == False:
                PathData.append(PathDataSet[Data])
            elif os.path.isdir(filePath + '\\' + PathDataSet[Data]) == True:
                pass
        for got in range(len(PathData)):# 将拿到的所有文件放到数组中进行右边窗口赋值。
            gosData = QStandardItem(PathData[got])
            PathDataName.setChild(got, gosData)

    def path_DoubleClicked(self, Qmodelidx):
        indexItem = self.model02.index(Qmodelidx.row(), 0, Qmodelidx.parent())
        self.file_Name = self.model02.data(indexItem)
        self.file_Path = self.model01.filePath(self.treeView1.currentIndex())
        self.open_Pic()
        if self.ischinese == False:
            self.start_Rec()

    def path_Clicked(self, Qmodelidx):
        indexItem = self.model02.index(Qmodelidx.row(), 0, Qmodelidx.parent())
        self.file_Name = self.model02.data(indexItem)
        self.file_Path = self.model01.filePath(self.treeView1.currentIndex())

    def is_chinese(self,word):
        for ch in word:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    def open_Pic(self):
        self.scan_label.setPixmap(QPixmap('one.png').scaled(QSize(150, 70)))
        self.txt_label.setText("plate_Number")
        if self.file_Path!=None and self.file_Name!=0:
            if self.file_Name[-3:]=='jpg' or self.file_Name[-3:]=='JPG':
                self.ischinese = self.is_chinese(self.file_Name)   #判斷檔名是否有中文
                if self.ischinese == False:
                    self.textEdit.append('路徑為：'+self.file_Path + '/' + self.file_Name)
                    self.textEdit.append('---------------')
                    self.textEdit.moveCursor(QTextCursor.End)
                    self.car= cv2.imread(self.file_Path+'/'+self.file_Name)
                    self.start_bt.setEnabled(True)
                    self.car_refreshShow()
                else:
                    self.car = np.array([1])
                    self.car_refreshShow()
                    self.textEdit.append("檔案不能有中文名")
                    self.textEdit.append('---------------')
                    self.textEdit.moveCursor(QTextCursor.End)
            else:
                self.car = np.array([1])
                self.car_refreshShow()
                self.textEdit.append("請打開jpg格式的圖片")
                self.textEdit.append('---------------')
                self.textEdit.moveCursor(QTextCursor.End)
        else:
            self.textEdit.append("路劲错误")
            self.textEdit.append('---------------')
            self.textEdit.moveCursor(QTextCursor.End)

    def car_refreshShow(self):
        height, width, channel =  self.car.shape
        bytesPerline = 3 * width
        self.qImg = QImage( self.car, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ori_label.setPixmap(QPixmap.fromImage(self.qImg).scaled(QSize(600,425)))
        self.ori_label.setAlignment(Qt.AlignCenter)#居中显示

    def ctpn_detect(self,car,new_plate,isfile_flag,isplate_flag):
        if isfile_flag == False:
            self.textEdit.append("CTPN模型丟失！")
            self.bt_open()
        elif isplate_flag == False or len(np.unique(new_plate)) == 1:
            self.textEdit.append("無法識別車牌！")
            self.bt_open()
        else:
            self.new_plate = new_plate
            self.car = car
            self.plate_refreshShow()
            height, width, channel = self.car.shape
            bytesPerline = 3 * width
            self.qImg = QImage(self.car, width, height, bytesPerline, QImage.Format_BGR888)
            self.ori_label.setPixmap(QPixmap.fromImage(self.qImg).scaled(QSize(600, 425)))
            self.ori_label.setAlignment(Qt.AlignCenter)  # 居中显示
            self.textEdit.append('CTPN完成')
            if self.rbt_Ocrtype == "pyocr":
                self.pyocr_ocr()
            elif self.rbt_Ocrtype == "easyocr":
                self.easyocr_ocr()
        self.bt_open()
        self.textEdit.append('---------------')
        self.textEdit.moveCursor(QTextCursor.End)


    def easyocr_ocr(self):
        self.textEdit.append('easyocr識別中')
        reader = easyocr.Reader(['en'])
        result = reader.readtext(self.new_plate)
        for i in result:
            self.txt= i[1]
            self.txt_label.setText(self.txt)
        self.textEdit.append('识别结束')
        self.save_bt.setEnabled(True)



    def pyocr_ocr(self):
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            self.textEdit.append("無法找到Tesseract-OCR")
            self.textEdit.append('---------------')
            self.textEdit.moveCursor(QTextCursor.End)
        else:
            self.textEdit.append('pyocr识别中')
            tool = tools[0]
            self.txt = tool.image_to_string(
                image=Image.fromarray(cv2.cvtColor(self.new_plate, cv2.COLOR_BGR2RGB)),
                builder=pyocr.builders.TextBuilder()
            )
        self.txt_label.setText(self.txt)
        self.textEdit.append('识别结束')
        self.save_bt.setEnabled(True)


    def bt_close(self):
        self.model_haar.setEnabled(False)
        self.model_ctpn.setEnabled(False)
        self.model_yolo.setEnabled(False)
        self.py_ocr.setEnabled(False)
        self.easy_ocr.setEnabled(False)
        self.treeView1.setEnabled(False)
        self.treeView2.setEnabled(False)
        self.start_bt.setEnabled(False)
        self.save_bt.setEnabled(False)
        self.open_bt.setEnabled(False)

    def bt_open(self):
        self.model_haar.setEnabled(True)
        self.model_ctpn.setEnabled(True)
        self.model_yolo.setEnabled(True)
        self.py_ocr.setEnabled(True)
        self.easy_ocr.setEnabled(True)
        self.treeView1.setEnabled(True)
        self.treeView2.setEnabled(True)
        self.start_bt.setEnabled(True)
        self.open_bt.setEnabled(True)

    def haar_detect(self,car,new_plate,isfile_flag,isplate_flag):
        if isfile_flag==False:
            self.textEdit.append("Haar模型丟失！")
            self.bt_open()
        elif isplate_flag==False or len(np.unique(new_plate)) == 1:
            self.textEdit.append("無法識別車牌！")
            self.bt_open()
        else:
            self.new_plate = new_plate
            self.car = car
            self.plate_refreshShow()
            height, width, channel = self.car.shape
            bytesPerline = 3 * width
            self.qImg = QImage(self.car, width, height, bytesPerline, QImage.Format_BGR888)
            self.ori_label.setPixmap(QPixmap.fromImage(self.qImg).scaled(QSize(600, 425)))
            self.ori_label.setAlignment(Qt.AlignCenter)  # 居中显示
            self.textEdit.append('Haar完成')
            if self.rbt_Ocrtype == "pyocr":
                self.pyocr_ocr()
            elif self.rbt_Ocrtype == "easyocr":
                self.easyocr_ocr()
        self.bt_open()
        self.textEdit.append('---------------')
        self.textEdit.moveCursor(QTextCursor.End)

    def yolo_detect(self,car,new_plate,isfile_flag,isplate_flag):
        if isfile_flag == False:
            self.textEdit.append("YOLO配置文件丟失！")
            self.bt_open()
        elif isplate_flag == False or len(np.unique(new_plate)) == 1:
            self.textEdit.append("無法識別車牌！")
            self.bt_open()
        else:
            self.new_plate = new_plate
            self.car = car
            self.plate_refreshShow()
            height, width, channel = self.car.shape
            bytesPerline = 3 * width
            self.qImg = QImage(self.car, width, height, bytesPerline, QImage.Format_BGR888)
            self.ori_label.setPixmap(QPixmap.fromImage(self.qImg).scaled(QSize(600, 425)))
            self.ori_label.setAlignment(Qt.AlignCenter)  # 居中显示
            self.textEdit.append('YOLO完成')
            if self.rbt_Ocrtype == "pyocr":
                self.pyocr_ocr()
            elif self.rbt_Ocrtype == "easyocr":
                self.easyocr_ocr()
        self.bt_open()
        self.textEdit.append('---------------')
        self.textEdit.moveCursor(QTextCursor.End)

    def start_Rec(self):
        if self.file_Path != None and self.file_Name != None and (self.file_Name[-3:] == 'jpg' or self.file_Name[-3:] == 'JPG'):
            self.new_plate=np.array([1])
            self.open_Pic()
            if self.rbt_Modeltype=="haar":
                self.bt_close()
                self.textEdit.append('HAAR车牌侦测中。。。。')
                self.Haar_threads = Haar_Thread(self.car)
                self.Haar_threads._signal.connect(self.haar_detect)
                self.Haar_threads.start()
            elif self.rbt_Modeltype=="ctpn":
                self.bt_close()
                self.textEdit.append('CTPN车牌侦测中。。。。')
                self.Ctpn_threads = Ctpn_Thread(self.car)
                self.Ctpn_threads._signal.connect(self.ctpn_detect)
                self.Ctpn_threads.start()
            elif self.rbt_Modeltype=="yolo":
                self.bt_close()
                self.textEdit.append('YOLO车牌侦测中。。。。')
                self.Yolo_threads = Yolo_Thread(self.car)
                self.Yolo_threads._signal.connect(self.yolo_detect)
                self.Yolo_threads.start()



    def plate_refreshShow(self):
        self.new_plate = cv2.cvtColor(np.asarray(self.new_plate), cv2.COLOR_RGB2BGR)
        height, width, channel =  self.new_plate.shape
        bytesPerline = 3 * width
        if self.rbt_Modeltype == "ctpn":
            self.qImg = QImage(self.new_plate, width, height, bytesPerline, QImage.Format_BGR888).rgbSwapped()
        else:
            self.qImg = QImage( self.new_plate, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.scan_label.setPixmap(QPixmap.fromImage(self.qImg).scaled(QSize(250,120)))
        self.scan_label.setAlignment(Qt.AlignCenter)#居中显示

    def saveSlot(self):
        if self.new_plate is None:
            return
        else:
            cv2.imwrite(self.file_Name[:-4]+'_plate.jpg', self.new_plate)
            infoBox = QMessageBox(self)  ##Message Box that doesn't run
            infoBox.setIcon(QMessageBox.Information)
            infoBox.setStandardButtons(QMessageBox.Ok)
            infoBox.setWindowTitle("Information")
            infoBox.setText(u"保存完成！(2秒自动关闭)")
            infoBox.button(QMessageBox.Ok).animateClick(3 * 1000)  # 3秒自动关闭
            infoBox.exec_()

if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    w.resize(1280,720)
    w.show()
    sys.exit(a.exec_())

