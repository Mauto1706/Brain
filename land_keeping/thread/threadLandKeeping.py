import cv2
import numpy as np
import threading
import time
import math
import matplotlib.pyplot as plt
import base64
from collections import deque
from src.templates.threadwithstop import ThreadWithStop  # Import lớp ThreadWithStop
from src.utils.messages.allMessages import(
    SteeringAngleFromLandKeeping
    serialCamera
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber

class ThreadLandKeeping(ThreadWithStop):
    def __init__(self, cap, queuesList, steering_angles_queue, logger=None, debugger=False):
        super(ThreadLandKeeping, self).__init__()
        #self.cap = cap   #Video capture object
        self.queuesList = queuesList
        #self.steering_angles_queue = steering_angles_queue  # Queue to store steering angles
        self.logger = logger
        self.debugger = debugger
        self.steering_angles = deque(maxlen=5)
        self.subcriber()
        
        self.sendSteeringAngle = messageHandleSender(self.queuesList, SteeringAngleFromLandKeeping)
    
    def subcriber(self):
        self.image = messageHandlerSubscriber(self.queuesList, serialCamera, "lastonly", True)
    
    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                # Đảm bảo line là mảng 1 chiều có 4 phần tử
                if not isinstance(line, np.ndarray):
                    line = np.array(line)
                if line.ndim == 2 and line.shape[0] == 1:
                    line = line.flatten()
                
                # Kiểm tra xem line có đủ 4 phần tử không
                if len(line) == 4:
                    x1, y1, x2, y2 = line
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                else:
                    print("Đoạn thẳng không hợp lệ:", line) # thay bang debug

        return line_image

    def average_slope_intercept(self, frame, line_segments):
        """Kết hợp các đoạn đường thành một hoặc hai đường làn."""
        lane_lines = []
        if line_segments is None:
            return lane_lines

        height, width, _ = frame.shape
        left_fit = []
        right_fit = []

        boundary = 1/3
        left_region_boundary = width * 1/2 # left lane line segment should be on left 2/3 of the screen
        right_region_boundary = width * 1/2 # right lane line segment should be on left 2/3 of the screen

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            lane_lines.append(self.make_points(frame, left_fit_average))

        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            lane_lines.append(self.make_points(frame, right_fit_average))

        return lane_lines
    def make_points(frame, line):       
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height  # bottom of the frame
        y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]

    def detect_edges(self, img):
        """Phát hiện các cạnh trong ảnh sử dụng Canny Edge Detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = 7
        blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        edges = cv2.Canny(blur, 100, 200)
        return edges

    def region_of_interest(self, canny):
        """Xác định khu vực quan tâm trong ảnh (mask khu vực chứa làn đường)."""
        height = canny.shape[0] #first dimension value
        width = canny.shape[1] #second dimension value
        mask = np.zeros_like(canny) #mask initialization
        shape = np.array([[(50, height), (width-50, height), (width-50, 300), (50,300)]], np.int32) #polygon 30
        cv2.fillPoly(mask, shape, 255) #mask with polygon size
        masked_image = cv2.bitwise_and(canny, mask) #final result
        #masked_resized_image = cv2.resize(masked_image, (400, 400))  
        #cv2.imshow("Canny mainlane", masked_resized_image)
        #cv2.waitKey(1)
        return masked_image

    def detect_line_segments(self, cropped_edges):
        rho = 1
        theta = np.pi / 180
        min_threshold = 20
        line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, np.array([]), minLineLength=60, maxLineGap=200)
        return line_segments

    def get_steering_angle(self, frame, lane_lines):
        """Tính toán góc lái dựa trên các đường làn."""
        height, width, _ = frame.shape
        if len(lane_lines) == 2:
            left_x1, left_y1, left_x2, left_y2 = lane_lines[0][0]
            right_x1, right_y1, right_x2, right_y2 = lane_lines[1][0]
            slope_l=math.atan((left_x1-left_x2) / (left_y1-left_y2))
            slope_r=math.atan((right_x1-right_x2) / (right_y1-right_y2))
            slope_ldeg = int(slope_l * 180.0 / math.pi)
            steering_angle_left = slope_ldeg  
            slope_rdeg = int(slope_r * 180.0 / math.pi)
            steering_angle_right = slope_rdeg
            if left_x2 > right_x2: #horizontal line 
                if abs(steering_angle_left) <= abs(steering_angle_right):
                    x_offset = left_x2 - left_x1
                    y_offset = int(height / 2)
                elif abs(steering_angle_left) > abs(steering_angle_right):
                    x_offset = right_x2 - right_x1
                    y_offset = int(height / 2)
            else: #normal left line
                    mid = int(width / 2)
                    x_offset = (left_x2 + right_x2) / 2 - mid
                    y_offset = int(height / 2)
        elif len(lane_lines) == 1:
                    x1, _, x2, _ = lane_lines[0][0]
                    x_offset = x2 - x1
                    y_offset = int(height / 2)
        elif len(lane_lines) == 0:
            x_offset = 0
            y_offset = int(height / 2)     
        #angle_to_mid_radian = math.atan(x_offset / y_offset)
        alfa = 0.6
        angle_to_mid_radian =(1-alfa)*math.atan(x_offset/ y_offset)
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
        steering_angle = angle_to_mid_deg + 90
        angle = angle_to_mid_radian
        return steering_angle
   
    def display_heading_line(self, frame, steering_angle, line_color=(0, 255,0), line_width=5):
        heading_image = np.zeros_like(frame)
        height, width, _ = frame.shape
        steering_angle_radian = steering_angle / 180.0 * math.pi
        x1 = int(width / 2)
        y1 = height
        x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
        y2 = int(height / 1.75)
        cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
        return heading_image


    # def send_command(self, msgID, angle):
    #     """Gửi lệnh điều khiển góc lái."""    
    #     content = angle
    #     # Gửi lệnh điều khiển (ví dụ qua giao thức serial)
    #     print(f"Sending command: msgID={msgID}, angle={content}")

    def send_commands_from_queue(self):
        """Gửi các lệnh điều khiển từ hàng đợi."""
        while self._running:
            if self.steering_angles is not None:
                steering_angle = self.steering_angles[0]
                if steering_angle > 20:
                    #self.send_command(2, 22)
                    self.sendSteeringAngle(22)
                elif steering_angle < -25:
                    #self.send_command(2, -25)
                    self.sendSteeringAngle(-25)
                else:
                    self.sendSteeringAngle(int(steering_angle))
                self.steering_angles.popleft()
            time.sleep(0.01)

    def run(self):
        old_lines = None
        """Chạy thread và xử lý các khung hình từ video capture."""
        while self._running:
            img_bytes = self.image.receive()
            while img_bytes is not None:
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            #   _, frame = picam2.read()
                canny_image = detect_edges(frame)
                cropped_canny = region_of_interest(canny_image)
                lines  = detect_line_segments(cropped_canny)
                try:
                    averaged_lines = average_slope_intercept(frame, lines)
                    line_image = display_lines(frame, averaged_lines)
                    steering_angle = get_steering_angle(frame, averaged_lines)
                    heading_image = display_heading_line(line_image,steering_angle)
                    #combo_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
                    steering_angle = steering_angle - 90
                    steering_angles.append(int(steering_angle * 0.8))
                    steering_angles_1.append(steering_angle)
                    old_lines = averaged_lines
                    #resized_image = cv2.resize(combo_image, (400, 400))  
                    #cv2.imshow('result', resized_image)
                except:
                    if old_lines is not None:
                        line_image = display_lines(frame, old_lines)
                        combo_image = cv2.addWeighted(frame, 0.3, heading_image, 1, 1)
                        steering_angle = 90;
                      
            send_commands_from_queue() # Sử dụng ID 2 cho góc lái 
            time.sleep(0.5)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        

    def stop(self):
        """Dừng thread và đóng cửa sổ OpenCV."""
        #cv2.destroyAllWindows()
        super(ThreadLandKeeping, self).stop()

# # Tạo đối tượng thread và chạy
# cap = cv2.VideoCapture(0)
# steering_angles_queue = deque()  # Queue chứa các góc lái
# land_keeping_thread = ThreadLandKeeping(cap, steering_angles_queue)
# land_keeping_thread.start()
