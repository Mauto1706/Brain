# threadBienBao.py

from src.templates.threadwithstop import ThreadWithStop
import numpy as np
import cv2
import time
from tensorflow.lite.python.interpreter import Interpreter
from src.utils.messages.allMessages import (
    mainCamera,
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber

class threadTrafficSign(ThreadWithStop):
    def __init__(self, queueList, logger, debugging=False):
        super(threadTrafficSign, self).__init__()
        self.logger = logger
        self.debugging = debugging
        self.queuesList = queueList

        self.model_path = "detect.tflite"
        self.label_path = "labelmap.txt"
        self.min_confidence = 0.5
        self.current_command = None
        self.cap = cv2.VideoCapture("path_to_video.mp4")  # You can use 0 for live camera

        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.float_input = (self.input_details[0]['dtype'] == np.float32)

        self.subcribe()

        with open(self.label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def subcribe(self)
            self.recordSubscriber = messageHandlerSubscriber(self.queuesList, Record, "lastOnly", True)

    def send_command_signal(self, msgID, angle):
        if angle != self.current_command:
            self.current_command = angle
            if self.debugging:
                print("Sending command:", {"msgID": msgID, "angle": angle})
            self.queuesList["Critical"].put(
                {
                    "Owner": "BienBao",
                    "msgID": msgID,
                    "msgType": "AngleCommand",
                    "msgValue": angle,
                }
            )

    def detect_objects(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)

        if self.float_input:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        signal = 'none'
        distance = 0

        for i in range(len(scores)):
            if scores[i] > self.min_confidence:
                ymin = int(max(1, boxes[i][0] * imH))
                xmin = int(max(1, boxes[i][1] * imW))
                ymax = int(min(imH, boxes[i][2] * imH))
                xmax = int(min(imW, boxes[i][3] * imW))

                label = self.labels[int(classes[i])]
                distance = int((-0.4412) * (xmax - xmin) + 63.9706)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(frame, f'{label}, Distance:{distance}', (xmin - 150, ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (208, 224, 64), 2)
                signal = classes[i]
                break

        return distance, signal

    def run(self):
        cv2.startWindowThread()
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            distance_output, signal2 = self.detect_objects(frame)
            resized_image = cv2.resize(frame, (400, 400))
            cv2.imshow("BienBao Detector", resized_image)

            if signal2 == "none":
                self.send_command_signal(1, 10)
            elif signal2 == 8.0 and distance_output < 20:
                self.send_command_signal(1, 0)
                time.sleep(5)
                self.send_command_signal(1, 10)
                time.sleep(2)
            elif signal2 == 0.0 and distance_output < -30:
                self.send_command_signal(1, 0)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
        # self.cap.release()
        # cv2.destroyAllWindows()

    def stop(self)
        super(threadTrafficSign, self).stop()
