# CheckRecognition_TW
### 使用CTPN、HAAR和YOLO来侦测台湾车牌。
### 使用Pyocr和EasyOCR来识别车牌。
------ 
## 文件说明  
1. car_demo --用于测试的车牌图片   
2. darknet_model --已训练的YOLO模型  
   模型链接：[阿里云盘](https://www.aliyundrive.com/s/yaBp4S9MoXE)  [Google Drive](https://drive.google.com/drive/folders/1DY5oTY0MvCjW1xhs_0ALIErkF8OtFa7f?usp=sharing)  
   来源：https://www.kaggle.com/datasets/achrafkhazri/yolo-weights-for-licence-plate-detector
3. haar_model --已训练的HAAR模型  
   模型链接：[阿里云盘](https://www.aliyundrive.com/s/UPwEc9s4Yra)  [Google Drive](https://drive.google.com/drive/folders/1lq5_93EOAEBuz4KEWimWRW6XKcMRatVr?usp=sharing)  
4. CTPN_model --已训练的CTPN模型    
   模型链接：[阿里云盘](https://www.aliyundrive.com/s/Lg6uPvgAkoL)  [Google Drive](https://drive.google.com/drive/folders/1U0YqsXZVecaVGMddIprCO_eh87LF7yeb?usp=sharing)  
5. nets、utils --CTPN 模型    
6. pic --预载图片   
7. CarMain.py --主程式  
8. CTPN_detect.py --CTPN侦测程式 
9. HAAR_detect.py --HAAR侦测程式  
10. YOLO_detect.py --YOLO侦测程式    
11. CarMain.spec --pyinstaller 生成 exe(小黑 pyinstaller RecognitionLoad.spec)    
------ 
## demo
    CarMain.py
