import sys
import cv2
import numpy as np
import platform
import random
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QBrush, QPainterPath, QPen
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QThread, pyqtSignal


class FaceDetectThread(QThread):
    face_data_signal = pyqtSignal(float, float, bool)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

        self.scale_factor = 1.1
        self.min_neighbors = 4
        self.min_size = (80, 80)

        self.last_face_rect = None
        self.roi_expand = 2.0
        self.lost_frame_count = 0
        self.max_lost_frames = 20
        self.detect_confidence = 0
        self.min_confidence = 2

        self.max_history_frames = 10
        self.normalized_x_history = []
        self.normalized_y_history = []
        self.weights = np.linspace(0.05, 0.2, self.max_history_frames)
        self.weights = self.weights / self.weights.sum()

        self.expand_search_frames = 0
        self.cap = None

    def run(self):
        try:
            cap_backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(0, cap_backend)

            if not self.cap.isOpened():
                print("错误：无法打开摄像头")
                self.face_data_signal.emit(0, 0, False)
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.msleep(10)
                    continue

                frame_height, frame_width = frame.shape[:2]
                if frame_width == 0 or frame_height == 0:
                    self.msleep(10)
                    continue

                detect_scale = 0.5
                small_frame = cv2.resize(
                    frame,
                    (int(frame_width * detect_scale), int(frame_height * detect_scale))
                )
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                gray = cv2.equalizeHist(gray)

                faces = []
                if self.last_face_rect is not None and self.lost_frame_count == 0:
                    faces = self._detect_with_roi(gray, frame_width, frame_height, detect_scale)
                    if len(faces) == 0:
                        faces = self._detect_full_image(gray, frame_width, frame_height, detect_scale)
                elif self.lost_frame_count > 0 and self.lost_frame_count < self.max_lost_frames:
                    faces = self._detect_with_roi(gray, frame_width, frame_height, detect_scale)
                    if len(faces) == 0:
                        faces = self._detect_full_image(gray, frame_width, frame_height, detect_scale)
                else:
                    faces = self._detect_full_image(gray, frame_width, frame_height, detect_scale)

                if len(faces) > 0:
                    self.detect_confidence = min(self.detect_confidence + 1, 10)
                    self.lost_frame_count = 0
                    self.expand_search_frames = 0

                    x, y, w, h = faces[0]
                    self.last_face_rect = (x, y, w, h)
                    face_center_x = x + w / 2
                    face_center_y = y + h / 2

                    normalized_x = -(face_center_x - frame_width / 2) / (frame_width / 2)
                    normalized_y = (face_center_y - frame_height / 2) / (frame_height / 2)

                    dead_zone = 0.03
                    normalized_x = 0 if abs(normalized_x) < dead_zone else normalized_x
                    normalized_y = 0 if abs(normalized_y) < dead_zone else normalized_y

                    self.normalized_x_history.append(normalized_x)
                    self.normalized_y_history.append(normalized_y)
                    if len(self.normalized_x_history) > self.max_history_frames:
                        self.normalized_x_history.pop(0)
                        self.normalized_y_history.pop(0)

                    if self.detect_confidence >= self.min_confidence:
                        n = len(self.normalized_x_history)
                        w_slice = self.weights[-n:] if n <= self.max_history_frames else self.weights
                        w_slice = w_slice / w_slice.sum()
                        avg_x = np.average(self.normalized_x_history, weights=w_slice)
                        avg_y = np.average(self.normalized_y_history, weights=w_slice)
                        if np.isfinite(avg_x) and np.isfinite(avg_y):
                            self.face_data_signal.emit(float(avg_x), float(avg_y), True)
                        else:
                            self.face_data_signal.emit(0, 0, False)
                    else:
                        self.face_data_signal.emit(0, 0, False)

                else:
                    self.lost_frame_count += 1

                    if self.lost_frame_count >= self.max_lost_frames:
                        self.detect_confidence = 0
                        self.last_face_rect = None
                        self.normalized_x_history.clear()
                        self.normalized_y_history.clear()
                        self.face_data_signal.emit(0, 0, False)
                    else:
                        if self.detect_confidence >= self.min_confidence and self.last_face_rect:
                            x, y, w, h = self.last_face_rect
                            face_center_x = x + w / 2
                            face_center_y = y + h / 2
                            normalized_x = -(face_center_x - frame_width / 2) / (frame_width / 2)
                            normalized_y = (face_center_y - frame_height / 2) / (frame_height / 2)
                            self.face_data_signal.emit(float(normalized_x), float(normalized_y), True)
                        else:
                            self.face_data_signal.emit(0, 0, False)

        except Exception as e:
            print(f"检测线程错误：{e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()

    def _detect_with_roi(self, gray, frame_width, frame_height, detect_scale):
        if self.last_face_rect is None:
            return []
        x, y, w, h = self.last_face_rect
        expand = self.roi_expand
        x1 = max(0, int((x - w * (expand - 1) / 2) * detect_scale))
        y1 = max(0, int((y - h * (expand - 1) / 2) * detect_scale))
        x2 = min(int(frame_width * detect_scale), int((x + w + w * (expand - 1) / 2) * detect_scale))
        y2 = min(int(frame_height * detect_scale), int((y + h + h * (expand - 1) / 2) * detect_scale))

        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size == 0 or roi_gray.shape[0] < 20 or roi_gray.shape[1] < 20:
            return []

        roi_faces = self.face_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(int(self.min_size[0] * detect_scale), int(self.min_size[1] * detect_scale))
        )

        faces = []
        for (rx, ry, rw, rh) in roi_faces:
            faces.append([
                (rx + x1) / detect_scale,
                (ry + y1) / detect_scale,
                rw / detect_scale,
                rh / detect_scale
            ])
        return faces

    def _detect_full_image(self, gray, frame_width, frame_height, detect_scale):
        detected = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(int(self.min_size[0] * detect_scale), int(self.min_size[1] * detect_scale))
        )
        if len(detected) == 0:
            return []
        return [[x / detect_scale, y / detect_scale, w / detect_scale, h / detect_scale]
                for (x, y, w, h) in detected]

    def stop(self):
        self.is_running = False
        self.wait()


class FloatingHeart:
    """浮动爱心粒子类"""
    def __init__(self, x, y, size, speed, color_variant=0):
        self.x = x
        self.y = y
        self.start_y = y
        self.size = size
        self.speed = speed
        self.opacity = 1.0
        self.sway_offset = random.uniform(0, 360)  # 左右摆动相位
        self.sway_amplitude = random.uniform(10, 25)  # 摆动幅度
        self.rotation = random.uniform(-15, 15)  # 旋转角度
        self.color_variant = color_variant  # 颜色变体（0-2）
        
    def update(self):
        """更新爱心位置"""
        self.y -= self.speed
        # 左右摆动效果
        self.sway_offset += 3
        # 渐隐效果
        traveled = self.start_y - self.y
        if traveled > 100:
            self.opacity = max(0, 1.0 - (traveled - 100) / 150)
        
    def get_x_with_sway(self):
        """获取带摆动的 X 坐标"""
        return self.x + np.sin(np.radians(self.sway_offset)) * self.sway_amplitude
    
    def is_alive(self):
        """判断爱心是否还活着"""
        return self.y > -50 and self.opacity > 0


class StackChanAvatar(QWidget):
    # ✅ 表情持续时间配置（毫秒）
    EMOTION_DURATIONS = {
        'happy': 1000,      # 开心 1 秒
        'angry': 1000,      # 生气 1 秒
        'love': 7000,       # 爱心 7 秒
        'wink': 800,        # Wink 一次眨眼约 0.8 秒
        'snowflake': 5000   # 雪花屏 5 秒
    }

    def __init__(self):
        super().__init__()
        self.initUI()
        self.initFaceThread()

        self.eye_radius = 35
        self.base_eye_center_left = QPointF(150, 150)
        self.base_eye_center_right = QPointF(250, 150)
        self.eye_center_left = QPointF(150, 150)
        self.eye_center_right = QPointF(250, 150)

        self.face_offset_x = 0
        self.face_offset_y = 0
        self.target_face_x = 0
        self.target_face_y = 0
        self.base_face_rect = QRectF(80, 50, 240, 200)
        self.face_rect = QRectF(80, 50, 240, 200)

        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.blink)
        self.blink_timer.start(4000)
        self.is_blinking = False
        self.blink_step = 0

        self.face_detected = False
        self.eye_move_scale = 1.5

        self.ui_lost_count = 0
        self.ui_max_lost = 10

        self.current_emotion = 'normal'
        self.emotion_timer = QTimer(self)
        self.emotion_timer.timeout.connect(self.updateEmotion)
        self.emotion_step = 0

        # ✅ 随机表情定时器（每 10 秒触发）
        self.random_emotion_timer = QTimer(self)
        self.random_emotion_timer.timeout.connect(self.randomEmotion)
        self.random_emotion_timer.start(10000)

        # ✅ 表情持续时间定时器
        self.emotion_duration_timer = QTimer(self)
        self.emotion_duration_timer.timeout.connect(self.restoreNormalEmotion)
        self.emotion_duration_timer.setSingleShot(True)

        # ✅ 雪花屏噪点动画定时器
        self.snowflake_noise_timer = QTimer(self)
        self.snowflake_noise_timer.timeout.connect(self.updateSnowflakeNoise)
        self.snowflake_noise_timer.setSingleShot(False)

        # 雪花屏状态
        self.snowflake_intensity = 0.5
        
        # 雪花屏噪点数据
        self.snowflake_noise = None
        self.snowflake_intensity = 0.5

        # ✅ 浮动爱心系统
        self.floating_hearts = []
        self.heart_spawn_timer = QTimer(self)
        self.heart_spawn_timer.timeout.connect(self.spawnFloatingHeart)
        self.heart_animation_timer = QTimer(self)
        self.heart_animation_timer.timeout.connect(self.updateFloatingHearts)

        # 手动控制模式标志
        self.manual_mode = False

    def initUI(self):
        self.setWindowTitle("StackChan AI Assistant - 浮动爱心版")
        self.setGeometry(100, 100, 400, 300)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #000000;")
        self.setFocusPolicy(Qt.StrongFocus)

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.updateUI)
        self.ui_timer.start(16)

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        self.setFocus()

    def initFaceThread(self):
        self.face_thread = FaceDetectThread()
        self.face_thread.face_data_signal.connect(self.on_face_data_received)
        self.face_thread.start()

    def on_face_data_received(self, norm_x, norm_y, is_detected):
        if not self.isVisible():
            return
        try:
            if is_detected:
                self.ui_lost_count = 0
                self.face_detected = True
                face_move_scale = 120
                self.target_face_x = norm_x * face_move_scale
                self.target_face_y = norm_y * face_move_scale
            else:
                self.ui_lost_count += 1
                if self.ui_lost_count >= self.ui_max_lost:
                    self.face_detected = False
                    self.target_face_x = 0
                    self.target_face_y = 0
        except Exception:
            pass

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_1:
            self.manual_mode = True
            self.setEmotion('normal')
        elif key == Qt.Key_2:
            self.manual_mode = True
            self.setEmotion('angry')
        elif key == Qt.Key_3:
            self.manual_mode = True
            self.setEmotion('happy')
        elif key == Qt.Key_4:
            self.manual_mode = True
            self.setEmotion('love')
        elif key == Qt.Key_5:
            self.manual_mode = True
            self.setEmotion('wink')
        elif key == Qt.Key_6:
            self.manual_mode = True
            self.setEmotion('snowflake')
        elif key == Qt.Key_A:
            self.manual_mode = not self.manual_mode
            if not self.manual_mode:
                self.setEmotion('normal')
            self.update()
        elif key == Qt.Key_Space:
            self.manual_mode = False
            self.setEmotion('normal')
        else:
            super().keyPressEvent(event)

    def setEmotion(self, emotion, duration=None):
        """设置表情，可指定持续时间"""
        self.current_emotion = emotion
        self.emotion_step = 0

        # 停止之前的定时器
        self.emotion_duration_timer.stop()
        self.snowflake_noise_timer.stop()
        self.heart_spawn_timer.stop()
        self.heart_animation_timer.stop()
        self.floating_hearts.clear()

        if emotion == 'wink':
            self.emotion_timer.start(80)
            self.blink_timer.stop()
            # Wink 只眨眼一次
            QTimer.singleShot(800, lambda: self.restoreNormalEmotion() if not self.manual_mode else None)
        elif emotion == 'snowflake':
            self.emotion_timer.stop()
            self.blink_timer.stop()
            # 初始化雪花屏噪点
            self.initSnowflakeNoise()
            # 启动噪点动画（每 50ms 刷新一次噪点）
            self.snowflake_noise_timer.start(50)
            # 5 秒后恢复正常
            QTimer.singleShot(5000, lambda: self.restoreNormalEmotion() if not self.manual_mode else None)
        elif emotion == 'love':
            self.emotion_timer.stop()
            self.blink_timer.stop()
            # ✅ 启动浮动爱心系统
            self.heart_spawn_timer.start(150)  # 每 150ms 生成一个爱心
            self.heart_animation_timer.start(30)  # 每 30ms 更新动画
            # 7 秒后恢复正常
            QTimer.singleShot(7000, lambda: self.restoreNormalEmotion() if not self.manual_mode else None)
        else:
            self.emotion_timer.stop()
            self.blink_timer.start(4000)

        # 设置持续时间（如果没有手动指定）
        if duration is None and emotion in self.EMOTION_DURATIONS:
            duration = self.EMOTION_DURATIONS[emotion]

        if duration and not self.manual_mode and emotion != 'wink' and emotion != 'snowflake' and emotion != 'love':
            self.emotion_duration_timer.start(duration)

        self.update()

    def spawnFloatingHeart(self):
        """生成浮动爱心"""
        if self.current_emotion != 'love':
            return
            
        width = self.width()
        height = self.height()
        
        # 随机生成爱心参数
        x = random.randint(20, width - 20)
        y = height + 20  # 从底部生成
        size = random.randint(15, 35)
        speed = random.uniform(1.5, 3.5)
        color_variant = random.randint(0, 2)
        
        heart = FloatingHeart(x, y, size, speed, color_variant)
        self.floating_hearts.append(heart)
        
        # 限制爱心数量，防止性能问题
        if len(self.floating_hearts) > 30:
            self.floating_hearts.pop(0)

    def updateFloatingHearts(self):
        """更新所有浮动爱心"""
        # 更新所有爱心位置
        for heart in self.floating_hearts:
            heart.update()
        
        # 移除消失的爱心
        self.floating_hearts = [h for h in self.floating_hearts if h.is_alive()]
        
        self.update()

    def initSnowflakeNoise(self):
        """初始化雪花屏噪点数据"""
        width = self.width()
        height = self.height()
        if width > 0 and height > 0:
            # 生成随机噪点图案（灰度值）
            self.snowflake_noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        self.snowflake_intensity = 0.5

    def updateSnowflakeNoise(self):
        """更新雪花屏噪点（模拟电视雪花效果）"""
        if self.current_emotion == 'snowflake':
            width = self.width()
            height = self.height()
            if width > 0 and height > 0:
                # 每帧生成新的随机噪点
                self.snowflake_noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            self.update()

    def randomEmotion(self):
        """每 10 秒随机切换表情（仅在非手动模式下）"""
        if self.manual_mode:
            return

        # 如果当前正在展示表情，跳过本次随机
        if self.current_emotion != 'normal':
            return

        emotions = ['happy', 'angry', 'love', 'wink', 'snowflake']
        new_emotion = random.choice(emotions)
        self.setEmotion(new_emotion)

    def restoreNormalEmotion(self):
        """恢复为正常表情"""
        if self.manual_mode:
            return
        self.snowflake_noise_timer.stop()
        self.snowflake_noise = None
        self.heart_spawn_timer.stop()
        self.heart_animation_timer.stop()
        self.floating_hearts.clear()
        self.setEmotion('normal')



    def updateEmotion(self):
        self.emotion_step = (self.emotion_step + 1) % 8
        self.update()

    def blink(self):
        if not self.is_blinking:
            self.is_blinking = True
            self.blink_step = 0

    def updateUI(self):
        face_diff_x = abs(self.target_face_x - self.face_offset_x)
        face_diff_y = abs(self.target_face_y - self.face_offset_y)

        face_smooth = 0.12 if max(face_diff_x, face_diff_y) > 20 else 0.04

        self.face_offset_x = self.face_offset_x * (1 - face_smooth) + self.target_face_x * face_smooth
        self.face_offset_y = self.face_offset_y * (1 - face_smooth) + self.target_face_y * face_smooth

        self.face_rect = QRectF(
            self.base_face_rect.x() + self.face_offset_x,
            self.base_face_rect.y() + self.face_offset_y,
            self.base_face_rect.width(),
            self.base_face_rect.height()
        )

        self.eye_center_left = QPointF(
            self.base_eye_center_left.x() + self.face_offset_x * self.eye_move_scale,
            self.base_eye_center_left.y() + self.face_offset_y * self.eye_move_scale
        )
        self.eye_center_right = QPointF(
            self.base_eye_center_right.x() + self.face_offset_x * self.eye_move_scale,
            self.base_eye_center_right.y() + self.face_offset_y * self.eye_move_scale
        )

        if self.is_blinking:
            self.blink_step += 1
            if self.blink_step > 6:
                self.is_blinking = False
                self.blink_step = 0

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # ✅ 雪花屏效果：绘制全屏噪点
        if self.current_emotion == 'snowflake' and self.snowflake_noise is not None:
            self.drawSnowflakeScreen(painter)
        else:
            painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        try:
            # ✅ 爱心表情：先画浮动爱心背景，再画眼睛（眼睛在最上层）
            if self.current_emotion == 'love':
                self.drawFloatingHearts(painter)
                self.drawLoveEyes(painter)
            elif self.current_emotion != 'snowflake':
                self.drawEyes(painter)
            
            self.drawStatus(painter)
        except Exception as e:
            print(f"绘图错误：{e}")
        finally:
            painter.end()

    def drawFloatingHearts(self, painter):
        """绘制浮动爱心"""
        painter.save()
        try:
            for heart in self.floating_hearts:
                painter.save()
                
                # 设置透明度
                painter.setOpacity(heart.opacity)
                
                # 颜色变体（粉色系）
                colors = [
                    QColor(255, 100, 150),  # 深粉
                    QColor(255, 150, 180),  # 中粉
                    QColor(255, 180, 200),  # 浅粉
                ]
                color = colors[heart.color_variant]
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.NoPen)
                
                # 应用位置和旋转
                x = heart.get_x_with_sway()
                painter.translate(x, heart.y)
                painter.rotate(heart.rotation)
                
                # 绘制圆润爱心
                self._drawRoundedHeart(painter, 0, 0, heart.size)
                
                painter.restore()
        finally:
            painter.restore()

    def drawSnowflakeScreen(self, painter):
        """绘制电视雪花屏效果"""
        if self.snowflake_noise is None:
            return
            
        width = self.width()
        height = self.height()
        
        # 绘制噪点
        for y in range(0, height, 2):  # 隔行绘制提高性能
            for x in range(0, width, 2):
                if 0 <= y < self.snowflake_noise.shape[0] and 0 <= x < self.snowflake_noise.shape[1]:
                    gray_value = self.snowflake_noise[y, x]
                    
                    color = QColor(gray_value, gray_value, gray_value)
                    painter.setPen(color)
                    painter.drawPoint(x, y)
                    painter.drawPoint(x + 1, y)
                    painter.drawPoint(x, y + 1)
                    painter.drawPoint(x + 1, y + 1)

    def drawEyes(self, painter):
        painter.setPen(Qt.NoPen)
        if self.is_blinking:
            self.drawBlinkEyes(painter)
        elif self.current_emotion == 'angry':
            self.drawAngryEyes(painter)
        elif self.current_emotion == 'happy':
            self.drawHappyEyes(painter)
        elif self.current_emotion == 'wink':
            self.drawWinkEyes(painter)
        else:
            self.drawNormalEyes(painter)

    def drawNormalEyes(self, painter):
        painter.save()
        try:
            eye_size = int(self.eye_radius * 1.8)
            corner_radius = 10
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(Qt.NoPen)

            left_eye_rect = QRectF(
                self.eye_center_left.x() - eye_size / 2,
                self.eye_center_left.y() - eye_size / 2,
                eye_size, eye_size
            )
            right_eye_rect = QRectF(
                self.eye_center_right.x() - eye_size / 2,
                self.eye_center_right.y() - eye_size / 2,
                eye_size, eye_size
            )
            left_path = QPainterPath()
            left_path.addRoundedRect(left_eye_rect, corner_radius, corner_radius)
            painter.drawPath(left_path)

            right_path = QPainterPath()
            right_path.addRoundedRect(right_eye_rect, corner_radius, corner_radius)
            painter.drawPath(right_path)
        finally:
            painter.restore()

    def drawBlinkEyes(self, painter):
        painter.save()
        try:
            blink_factor = max(0.05, 1.0 - self.blink_step * 0.16)
            eye_size = int(self.eye_radius * 1.8)
            corner_radius = 10
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(Qt.NoPen)

            left_eye_rect = QRectF(
                self.eye_center_left.x() - eye_size / 2,
                self.eye_center_left.y() - eye_size / 2 * blink_factor,
                eye_size, eye_size * blink_factor
            )
            right_eye_rect = QRectF(
                self.eye_center_right.x() - eye_size / 2,
                self.eye_center_right.y() - eye_size / 2 * blink_factor,
                eye_size, eye_size * blink_factor
            )
            left_path = QPainterPath()
            left_path.addRoundedRect(left_eye_rect, corner_radius, corner_radius)
            painter.drawPath(left_path)

            right_path = QPainterPath()
            right_path.addRoundedRect(right_eye_rect, corner_radius, corner_radius)
            painter.drawPath(right_path)
        finally:
            painter.restore()

    def drawAngryEyes(self, painter):
        painter.save()
        try:
            eye_size = int(self.eye_radius * 1.8)
            corner_radius = 10
            brow_width = eye_size * 0.8
            brow_thickness = 6

            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(Qt.NoPen)

            left_eye_rect = QRectF(
                self.eye_center_left.x() - eye_size / 2,
                self.eye_center_left.y() - eye_size / 2,
                eye_size, eye_size
            )
            right_eye_rect = QRectF(
                self.eye_center_right.x() - eye_size / 2,
                self.eye_center_right.y() - eye_size / 2,
                eye_size, eye_size
            )
            left_path = QPainterPath()
            left_path.addRoundedRect(left_eye_rect, corner_radius, corner_radius)
            painter.drawPath(left_path)

            right_path = QPainterPath()
            right_path.addRoundedRect(right_eye_rect, corner_radius, corner_radius)
            painter.drawPath(right_path)

            painter.setPen(QPen(QColor(255, 255, 255), brow_thickness, Qt.SolidLine, Qt.RoundCap))

            left_brow = QPainterPath()
            left_brow.moveTo(
                self.eye_center_left.x() - brow_width / 2,
                self.eye_center_left.y() - eye_size / 2 - 15
            )
            left_brow.lineTo(
                self.eye_center_left.x() + brow_width / 2,
                self.eye_center_left.y() - eye_size / 2 - 5
            )
            painter.drawPath(left_brow)

            right_brow = QPainterPath()
            right_brow.moveTo(
                self.eye_center_right.x() - brow_width / 2,
                self.eye_center_right.y() - eye_size / 2 - 5
            )
            right_brow.lineTo(
                self.eye_center_right.x() + brow_width / 2,
                self.eye_center_right.y() - eye_size / 2 - 15
            )
            painter.drawPath(right_brow)
        finally:
            painter.restore()

    def drawHappyEyes(self, painter):
        painter.save()
        try:
            eye_size = int(self.eye_radius * 1.8)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255), 24, Qt.SolidLine, Qt.RoundCap))

            left_eye_rect = QRectF(
                self.eye_center_left.x() - eye_size / 2,
                self.eye_center_left.y() - eye_size / 2,
                eye_size, eye_size * 0.7
            )
            painter.drawArc(left_eye_rect, 0, 180 * 16)

            right_eye_rect = QRectF(
                self.eye_center_right.x() - eye_size / 2,
                self.eye_center_right.y() - eye_size / 2,
                eye_size, eye_size * 0.7
            )
            painter.drawArc(right_eye_rect, 0, 180 * 16)
        finally:
            painter.restore()

    def drawLoveEyes(self, painter):
        """绘制爱心眼睛（固定位置，不浮动）"""
        painter.save()
        try:
            heart_size = int(self.eye_radius * 2.0)
            painter.setBrush(QBrush(QColor(255, 50, 80)))
            painter.setPen(Qt.NoPen)
            self._drawRoundedHeart(painter, self.eye_center_left.x(), self.eye_center_left.y(), heart_size)
            self._drawRoundedHeart(painter, self.eye_center_right.x(), self.eye_center_right.y(), heart_size)
        finally:
            painter.restore()

    def _drawRoundedHeart(self, painter, center_x, center_y, size):
        """绘制经典爱心形状"""
        heart_path = QPainterPath()
        
        # 爱心底部尖点
        bottom_y = center_y + size * 0.5
        
        # 左半部分
        heart_path.moveTo(center_x, bottom_y)
        heart_path.cubicTo(
            center_x - size * 0.6, center_y,                # 控制点1
            center_x - size * 0.6, center_y - size * 0.6,    # 控制点2
            center_x, center_y - size * 0.35               # 顶部中心
        )
        
        # 右半部分
        heart_path.cubicTo(
            center_x + size * 0.6, center_y - size * 0.6,    # 控制点1
            center_x + size * 0.6, center_y,                # 控制点2
            center_x, bottom_y                                # 回到底部尖点
        )
        
        painter.drawPath(heart_path)

    def drawWinkEyes(self, painter):
        painter.save()
        try:
            eye_size = int(self.eye_radius * 1.8)
            corner_radius = 10
            wink_phase = self.emotion_step % 8

            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(Qt.NoPen)

            left_eye_rect = QRectF(
                self.eye_center_left.x() - eye_size / 2,
                self.eye_center_left.y() - eye_size / 2,
                eye_size, eye_size
            )
            left_path = QPainterPath()
            left_path.addRoundedRect(left_eye_rect, corner_radius, corner_radius)
            painter.drawPath(left_path)

            if wink_phase < 4:
                blink_factor = max(0.05, 1.0 - wink_phase * 0.25)
            else:
                blink_factor = max(0.05, (wink_phase - 4) * 0.25)

            right_eye_rect = QRectF(
                self.eye_center_right.x() - eye_size / 2,
                self.eye_center_right.y() - eye_size / 2 * blink_factor,
                eye_size,
                eye_size * blink_factor
            )
            right_path = QPainterPath()
            right_path.addRoundedRect(right_eye_rect, corner_radius, corner_radius)
            painter.drawPath(right_path)

        finally:
            painter.restore()

    def drawStatus(self, painter):
        painter.save()
        try:
            # 雪花屏时不显示状态（被噪点覆盖）
            if self.current_emotion == 'snowflake':
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(10, 20, "❄️ 雪花屏效果中...")
                return
                
            painter.setPen(Qt.NoPen)
            if self.face_detected:
                painter.setBrush(QBrush(QColor(0, 255, 100)))
            else:
                painter.setBrush(QBrush(QColor(255, 100, 100)))
            painter.drawEllipse(10, 10, 8, 8)

            painter.setPen(QColor(255, 255, 255))
            status_text = "跟踪中" if self.face_detected else "搜索中"
            painter.drawText(25, 17, status_text)

            emotion_map = {
                'normal': '正常',
                'angry': '生气',
                'happy': '开心',
                'love': '爱心',
                'wink': 'Wink',
                'snowflake': '雪花屏'
            }
            emotion_text = emotion_map.get(self.current_emotion, '正常')
            
            mode_text = "【手动】" if self.manual_mode else "【自动】"
            painter.drawText(25, 32, f"{mode_text} 表情：{emotion_text}")

            painter.setPen(QColor(150, 150, 150))
            painter.drawText(10, 265, "1-正常 2-生气 3-开心 4-爱心 5-Wink 6-雪花屏")
            painter.drawText(10, 280, "A-切换自动/手动  空格 - 重置为正常")
        finally:
            painter.restore()

    def closeEvent(self, event):
        self.face_thread.stop()
        self.blink_timer.stop()
        self.emotion_timer.stop()
        self.ui_timer.stop()
        self.random_emotion_timer.stop()
        self.emotion_duration_timer.stop()
        self.snowflake_noise_timer.stop()
        self.heart_spawn_timer.stop()
        self.heart_animation_timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = StackChanAvatar()
    ex.show()
    sys.exit(app.exec_())