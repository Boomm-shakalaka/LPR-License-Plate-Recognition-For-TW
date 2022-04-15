import cv2
import os
import sys
import numpy as np
class dark_detection:
    def __init__(self, car):
        self.car = car
        self.plate = None
        self.probability_minimum = 0.3
        self.threshold = 0.3

    def app_path(self):
        """Returns the base application path."""
        if hasattr(sys, 'frozen'):
            # Handles PyInstaller
            return os.path.dirname(sys.executable)  # 使用pyinstaller打包后的exe目录
        return os.path.dirname(__file__)  # 没打包前的py目录

    def darknet_detection(self):
        #self.car = cv2.resize(self.car, (600, 425))  # 尺寸300x225
        self.weights_file = self.app_path() + r'/darknet_model/carplate.weights'
        self.classes_file = self.app_path() + r'/darknet_model/classes.names'
        self.cfg_file = self.app_path() + r'/darknet_model/darknet-yolov3.cfg'
        self.weights_isfile = os.path.exists(self.weights_file)
        self.classes_isfile = os.path.exists(self.classes_file)
        self.cfg_isfile = os.path.exists(self.cfg_file)
        if  self.weights_isfile == False or self.classes_isfile == False or self.cfg_isfile == False:
            return
        network = cv2.dnn.readNetFromDarknet(self.cfg_file, self.weights_file)
        layers_names_all = network.getLayerNames()

        layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(self.car, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
        network.setInput(blob)
        output_from_network = network.forward(layers_names_output)
        np.random.seed(42)
        labels = open( self.classes_file).read()
        #colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        bounding_boxes = []
        confidences = []
        class_numbers = []
        h, w = self.car.shape[:2]

        for result in output_from_network:
            for detection in result:
                scores = detection[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > self.probability_minimum:
                    box_current = detection[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, self.probability_minimum, self.threshold)
        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                #colour_box_current = [int(j) for j in colours[class_numbers[i]]]

                self.car = cv2.rectangle(self.car, (x_min-5, y_min-5), (x_min + box_width+5, y_min + box_height+5),
                              (0, 255, 0), 1)

               # text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
                #cv2.putText(image_input, text_box_current, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX,
                          #  1.5, colour_box_current, 5)
                self.plate = self.car[y_min:y_min + box_height, x_min:x_min + box_width]

