# 导入操作系统相关模块
import os
import sys
import time
import json
import math
import requests
import pandas as pd
import cv2
import numpy as np
import threading
import base64
from queue import Queue
from datetime import datetime
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# 修复高分辨率屏 DPI 缩放问题
os.environ["QT_FONT_DPI"] = "96"

# 尝试导入YOLO - 使用本地模型，不下载权重
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("成功导入YOLO库")
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: 未找到YOLO库，目标检测功能不可用")

# 引入MQTT客户端库
import paho.mqtt.client as mqtt

# 导入PySide6 GUI组件
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtCharts import *
from PySide6.QtMultimedia import *
from PySide6.QtMultimediaWidgets import *
from PySide6.QtWebEngineWidgets import QWebEngineView

# 全局配置 - 更新时间和用户
CURRENT_USER = "12ljf"
CURRENT_DATE = "2025-08-25 08:47:50"

# MQTT配置
MQTT_BROKER = "47.107.36.182"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "USER001"
MQTT_USERNAME = "public"
MQTT_PASSWORD = "UQU92K77cpxc2Tm"
MQTT_TOPIC_PUBLISH = "USER001"    # 上位机发布控制命令
MQTT_TOPIC_SUBSCRIBE = "USER002"  # 上位机订阅传感器数据
MQTT_TOPIC_CAMERA = "USER002/camera"    # 摄像头数据主题
MQTT_TOPIC_CAMERA_NORMAL = "USER003"    # 普通摄像头数据主题
MQTT_TOPIC_STATUS = "USER002/status"    # 状态数据主题
MQTT_TOPIC_ALERTS = "USER002/alerts"    # 警报数据主题
MQTT_TOPIC_GPS = "USER002/gps"          # GPS数据主题

# MQTT连接设置
MQTT_CONNECT_TIMEOUT = 60
MQTT_KEEPALIVE = 60
MQTT_RECONNECT_DELAY_MIN = 1
MQTT_RECONNECT_DELAY_MAX = 30

# 模式和颜色配置 - 更新为科技蓝色调
GAIT_MODES = ["蠕动模式", "蜿蜒模式", "复位模式"]
GAIT_COLORS = {
    "蠕动模式": "#00E0E0",
    "蜿蜒模式": "#00AACC", 
    "复位模式": "#FF5555"
}
DIRECTIONS = {"前进": "↑", "后退": "↓", "左转": "←", "右转": "→", "复位": "↺"}

# YOLO配置 - 使用用户指定的完整路径
YOLO_MODEL_PATH = "C:\\Users\\li\\Desktop\\文件\\Snake Robot\\QT\\pygt5_learn-main\\pygt5_learn-main\\gui_tool_with_pyside6\\best.pt"
YOLO_CONFIDENCE = 0.6

# 字体配置 - 增加对中文的支持
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"  # Windows默认黑体
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/simsun.ttc"  # 备用宋体
if not os.path.exists(FONT_PATH):
    FONT_PATH = None  # 如果都不存在则使用默认
    print("警告: 未找到中文字体文件，将使用默认字体")

# 环境评估配置
ENVIRONMENT_LEVELS = ["优秀", "良好", "一般", "较差", "危险"]
ENVIRONMENT_COLORS = {
    "优秀": "#00E0E0",
    "良好": "#00CCFF",
    "一般": "#FFCC00",
    "较差": "#FF9500",
    "危险": "#FF5555"
}

# 警报阈值
ALERT_THRESHOLDS = {
    "voc_high": 250,
    "pressure_change": 2.0,
    "temperature_high": 50.0,
    "temperature_low": 0.0,
    "stability_warning": 0.7,
    "battery_low": 20.0
}

# 管道检测相关配置
PIPELINE_DEFECT_TYPES = ["裂缝", "腐蚀", "接头松动", "沉积物", "变形"]
PIPELINE_SEVERITY_COLORS = {
    "正常": "#00E0E0",
    "轻微": "#FFCC00",
    "中等": "#FF9500", 
    "严重": "#FF5555"
}

#====================== 增强的YOLO检测类 ======================

class YOLODetector:
    """YOLO目标检测器，基于ultralytics YOLO或YOLOv5"""
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_initialized = False
        self.names = None  # 类别名称
        self.device = 'cpu'
        self.last_detections = []  # 存储最后一次检测结果
        
        # 尝试初始化device
        try:
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                print(f"将使用GPU进行推理: {self.device}")
            else:
                print("未检测到可用GPU，将使用CPU进行推理")
        except ImportError:
            print("未找到PyTorch，将使用CPU进行推理")
            
        # 尝试加载字体
        self.font = None
        if FONT_PATH:
            try:
                # 尝试加载默认中文字体
                self.font = ImageFont.truetype(FONT_PATH, 30)  # 30点大小
                print(f"成功加载中文字体: {FONT_PATH}")
            except Exception as e:
                print(f"加载字体失败: {e}")
    
    def initialize(self):
        """初始化YOLO模型"""
        if not YOLO_AVAILABLE:
            print("YOLO库不可用，无法进行目标检测")
            return False
            
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            print(f"错误：本地YOLO模型文件不存在: {self.model_path}")
            return False
            
        try:
            print(f"正在加载YOLO模型: {self.model_path}")
            
            # 使用ultralytics YOLO加载模型
            self.model = YOLO(self.model_path)
            
            # 修正：确保名称字典正确加载
            if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                self.names = self.model.names
                print(f"成功加载类别名称: {self.names}")
            else:
                # 如果names不是字典，创建一个默认的名称映射
                self.names = {i: f"类别{i}" for i in range(10)}
                print(f"警告: 模型没有提供有效的类别名称，使用默认名称")
            
            # 禁用自动更新
            if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'args'):
                if hasattr(self.model.predictor.args, 'update'):
                    self.model.predictor.args.update = False
            
            self.is_initialized = True
            print(f"成功加载模型: {self.model_path}")
            print(f"检测类别: {self.names}")
            return True
            
        except Exception as e:
            print(f"模型初始化错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect(self, frame, confidence=YOLO_CONFIDENCE, draw=True):
        """执行目标检测"""
        if not self.is_initialized or self.model is None:
            if not self.initialize():
                return frame, []
        
        try:
            # 执行推理
            results = self.model(frame, conf=confidence, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                # 提取检测结果
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        # 提取坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # 修正：确保类别ID存在于names字典中
                        if cls_id in self.names:
                            cls_name = self.names[cls_id]
                        else:
                            cls_name = f"类别{cls_id}"
                        
                        detection = {
                            'class': cls_id,
                            'name': cls_name,
                            'confidence': conf,
                            'box': [int(x1), int(y1), int(x2), int(y2)],
                            'type': cls_name,  # 用于管道缺陷显示
                            'severity': self._determine_severity(conf),
                            'position': f"X:{int((x1+x2)/2)}, Y:{int((y1+y2)/2)}",
                            'size': f"{int(x2-x1)}x{int(y2-y1)}"
                        }
                        
                        detections.append(detection)
            
            # 保存最后一次检测结果
            self.last_detections = detections
            
            # 如果需要，在图像上绘制检测结果
            if draw and detections:
                frame = self._draw_detections_with_pil(frame, detections)
            
            return frame, detections
            
        except Exception as e:
            print(f"目标检测错误: {e}")
            import traceback
            traceback.print_exc()
            return frame, []
    
    def _determine_severity(self, confidence):
        """根据置信度确定缺陷严重性"""
        if confidence > 0.85:
            return "严重"
        elif confidence > 0.7:
            return "中等"
        elif confidence > 0.5:
            return "轻微"
        else:
            return "正常"
    
    def _draw_detections_with_pil(self, frame, detections):
        """使用PIL绘制检测结果，支持中文字符"""
        if not detections:
            return frame
        
        # 将OpenCV图像转换为PIL图像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 如果字体未加载，尝试使用默认字体
        if self.font is None:
            font_size = 20
            try:
                # 尝试加载任何可用字体
                self.font = ImageFont.load_default()
            except:
                # 如果无法加载字体，将使用OpenCV绘制
                return self._draw_detections(frame, detections)
        
        # 绘制每个检测结果
        for detection in detections:
            box = detection['box']
            cls_name = detection['name']
            severity = detection['severity']
            
            # 根据严重程度选择颜色
            if severity == '严重':
                color = (255, 0, 0)  # 红色
            elif severity == '中等':
                color = (255, 165, 0)  # 橙色
            elif severity == '轻微':
                color = (255, 255, 0)  # 黄色
            else:
                color = (0, 255, 0)  # 绿色
            
            # 绘制边界框
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)
            
            # 添加标签 - 移除置信度显示
            label = f"{cls_name}"  # 移除了置信度
            text_width, text_height = draw.textbbox((0, 0), label, font=self.font)[2:4]
            
            # 绘制标签背景
            draw.rectangle(
                [box[0], box[1] - text_height - 4, box[0] + text_width, box[1]],
                fill=color
            )
            
            # 绘制标签文字
            draw.text(
                (box[0], box[1] - text_height - 2),
                label,
                fill=(255, 255, 255),
                font=self.font
            )
        
        # 将PIL图像转换回OpenCV图像
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return opencv_image
    
    def _draw_detections(self, frame, detections):
        """使用OpenCV绘制检测结果（只支持ASCII字符，仅作为后备方法）"""
        for detection in detections:
            box = detection['box']
            cls_name = detection['name']
            severity = detection['severity']
            
            # 尝试使用ASCII字符替换中文
            cls_name = ''.join(c if ord(c) < 128 else '_' for c in cls_name)
            severity = ''.join(c if ord(c) < 128 else '_' for c in severity)
            
            # 根据严重程度选择颜色
            if severity == '严重' or severity == '____':
                color = (0, 0, 255)  # 红色
            elif severity == '中等' or severity == '__':
                color = (0, 165, 255)  # 橙色
            elif severity == '轻微' or severity == '__':
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 255, 0)  # 绿色
            
            # 绘制边界框
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # 添加标签 - 移除置信度显示
            label = f"{cls_name} ({severity})"  # 移除了置信度
            t_size = cv2.getTextSize(label, 0, 0.5, 1)[0]
            cv2.rectangle(
                frame, 
                (box[0], box[1] - t_size[1] - 3), 
                (box[0] + t_size[0], box[1]), 
                color, 
                -1
            )
            cv2.putText(
                frame, 
                label, 
                (box[0], box[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        return frame

#====================== 资源管理类 ======================

class ResourceManager:
    """资源管理器 - 使用单例模式管理硬件资源"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.cv_processors = {}
        self.ai_processors = {}
        self.resources_info = self._detect_resources()
        
        self._init_processors()
        
    def _detect_resources(self):
        """检测系统可用资源"""
        resources = {
            "gpu_available": False,
            "gpu_info": [],
            "cpu_count": os.cpu_count(),
            "camera_api": self._detect_camera_api(),
            "memory_available": self._get_available_memory()
        }
        
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                resources["gpu_available"] = True
                for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
                    dev_info = {
                        "index": i,
                        "name": f"GPU-{i}",
                        "compute_capability": "Unknown"
                    }
                    resources["gpu_info"].append(dev_info)
        except Exception:
            if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES']:
                resources["gpu_available"] = True
                resources["gpu_info"].append({
                    "index": 0,
                    "name": "GPU (from env)",
                    "compute_capability": "Unknown"
                })
            
        return resources
    
    def _detect_camera_api(self):
        """检测最佳摄像头API"""
        apis = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_AVFOUNDATION, "AVFoundation"),
            (cv2.CAP_ANY, "Auto")
        ]
        
        for api_id, api_name in apis:
            try:
                if sys.platform == 'win32' and api_id == cv2.CAP_DSHOW:
                    return api_id
                elif sys.platform == 'linux' and api_id == cv2.CAP_V4L2:
                    return api_id
                elif sys.platform == 'darwin' and api_id == cv2.CAP_AVFOUNDATION:
                    return api_id
            except Exception:
                continue
                
        return cv2.CAP_ANY
    
    def _get_available_memory(self):
        """获取可用内存（GB）"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 4.0
    
    def _init_processors(self):
        """初始化各种处理器"""
        self.cv_processors = {
            "cpu": CPUVideoProcessor(),
            "gpu": GPUVideoProcessor() if self.resources_info["gpu_available"] else CPUVideoProcessor()
        }
        
        self.ai_processors = {
            "cpu": CPUInferenceProcessor(),
            "gpu": GPUInferenceProcessor() if self.resources_info["gpu_available"] else CPUInferenceProcessor()
        }
    
    def get_video_processor(self):
        """获取最佳视频处理器"""
        if self.resources_info["gpu_available"]:
            return self.cv_processors["gpu"]
        return self.cv_processors["cpu"]
    
    def get_inference_processor(self):
        """获取最佳AI推理处理器"""
        if self.resources_info["gpu_available"]:
            return self.ai_processors["gpu"]
        return self.ai_processors["cpu"]
    
    def get_camera_api(self):
        """获取摄像头API"""
        return self.resources_info["camera_api"]
    
    def get_resources_summary(self):
        """获取资源概要"""
        summary = {
            "gpu_available": self.resources_info["gpu_available"],
            "cpu_count": self.resources_info["cpu_count"],
            "video_processor": "GPU" if self.resources_info["gpu_available"] else "CPU",
            "inference_processor": "GPU" if self.resources_info["gpu_available"] else "CPU"
        }
        return summary

#====================== 视频处理器 ======================

class VideoProcessor:
    """视频处理器接口"""
    def process_frame(self, frame):
        raise NotImplementedError
    
    def convert_to_qt(self, frame):
        raise NotImplementedError

class CPUVideoProcessor(VideoProcessor):
    """CPU视频处理器实现"""
    def process_frame(self, frame):
        if frame is None:
            return None
            
        try:
            if frame.size == 0:
                return None
                
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
            return frame
            
        except Exception as e:
            print(f"CPU视频处理错误: {e}")
            return frame
    
    def convert_to_qt(self, frame):
        """转换为Qt图像格式"""
        try:
            if frame is None or frame.size == 0:
                return None
                
            if len(frame.shape) == 3:
                height, width, channel = frame.shape
                if channel == 3:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * width
                    qt_image = QImage(rgb_frame.data, width, height, 
                                    bytes_per_line, QImage.Format_RGB888)
                    return qt_image
            
            return None
            
        except Exception as e:
            print(f"图像转换错误: {e}")
            return None

class GPUVideoProcessor(VideoProcessor):
    """GPU视频处理器实现"""
    def __init__(self):
        self.gpu_available = False
        try:
            self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            self.gpu_available = False
            
        if not self.gpu_available:
            print("警告: CUDA不可用，将使用CPU处理视频")
    
    def process_frame(self, frame):
        if frame is None or not self.gpu_available:
            cpu_processor = CPUVideoProcessor()
            return cpu_processor.process_frame(frame)
            
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                gpu_resized = cv2.cuda.resize(gpu_frame, 
                                            (int(w * scale), int(h * scale)))
                return gpu_resized.download()
            
            return gpu_frame.download()
            
        except Exception as e:
            print(f"GPU视频处理错误: {e}")
            cpu_processor = CPUVideoProcessor()
            return cpu_processor.process_frame(frame)
    
    def convert_to_qt(self, frame):
        cpu_processor = CPUVideoProcessor()
        return cpu_processor.convert_to_qt(frame)

#====================== AI推理处理器 ======================

class InferenceProcessor:
    """AI推理处理器接口"""
    def setup_model(self, model_path):
        raise NotImplementedError
    
    def infer(self, frame, confidence=0.5):
        raise NotImplementedError

class CPUInferenceProcessor(InferenceProcessor):
    """CPU推理处理器实现"""
    def setup_model(self, model_path):
        if not YOLO_AVAILABLE:
            print("YOLO库不可用，无法进行目标检测")
            return None
            
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 本地YOLO模型文件不存在: {model_path}")
            return None
            
        try:
            # 禁止自动下载，只使用本地模型
            model = YOLO(model_path, task='detect', device='cpu')
            print(f"成功加载模型: {model_path}")
            
            # 禁用自动更新
            if hasattr(model, 'predictor') and hasattr(model.predictor, 'args'):
                if hasattr(model.predictor.args, 'update'):
                    model.predictor.args.update = False
                
            return model
        except Exception as e:
            print(f"CPU模型设置错误: {e}")
            return None
    
    def infer(self, frame, model, confidence=0.25):
        if frame is None or model is None:
            return frame, []
            
        try:
            results = model(frame, conf=confidence, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'class': cls,
                            'name': result.names[cls],
                            'confidence': float(conf),
                            'box': [int(x1), int(y1), int(x2), int(y2)]
                        })
            
            return frame, detections
            
        except Exception as e:
            print(f"CPU推理错误: {e}")
            return frame, []

class GPUInferenceProcessor(InferenceProcessor):
    """GPU推理处理器实现"""
    def __init__(self):
        self.gpu_available = False
        
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.device = f"cuda:{torch.cuda.current_device()}"
                print(f"将使用GPU进行推理: {self.device}")
            else:
                self.device = "cpu"
                print("未检测到可用GPU，将使用CPU进行推理")
        except ImportError:
            self.device = "cpu"
            print("未找到PyTorch，将使用CPU进行推理")
            
    def setup_model(self, model_path):
        if not YOLO_AVAILABLE:
            print("YOLO库不可用，无法进行目标检测")
            return None
            
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 本地YOLO模型文件不存在: {model_path}")
            return None
            
        try:
            # 禁止自动下载，只使用本地模型
            model = YOLO(model_path, task='detect', device=self.device)
            print(f"成功加载模型到{self.device}: {model_path}")
            
            # 禁用自动更新
            if hasattr(model, 'predictor') and hasattr(model.predictor, 'args'):
                if hasattr(model.predictor.args, 'update'):
                    model.predictor.args.update = False
                    
            return model
        except Exception as e:
            print(f"GPU模型设置错误: {e}")
            try:
                # 尝试使用CPU
                print("尝试使用CPU加载模型...")
                return YOLO(model_path, task='detect', device='cpu')
            except Exception as e2:
                print(f"CPU备用模型设置错误: {e2}")
                return None
    
    def infer(self, frame, model, confidence=0.25):
        if frame is None or model is None:
            return frame, []
            
        if not self.gpu_available:
            cpu_processor = CPUInferenceProcessor()
            return cpu_processor.infer(frame, model, confidence)
            
        try:
            results = model(frame, conf=confidence, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'class': cls,
                            'name': result.names[cls],
                            'confidence': float(conf),
                            'box': [int(x1), int(y1), int(x2), int(y2)]
                        })
            
            return frame, detections
            
        except Exception as e:
            print(f"GPU推理错误: {e}")
            cpu_processor = CPUInferenceProcessor()
            return cpu_processor.infer(frame, model, confidence)

#====================== 多模态数据融合类 ======================

class MultiModalDataFusion:
    """多模态数据融合与分析类 - 增强管道检测分析"""
    def __init__(self):
        # 初始化数据容器
        self.sensor_data = {}
        self.sensor_history = {
            "temperature": deque(maxlen=200),
            "pressure": deque(maxlen=200),
            "air_quality": deque(maxlen=200),
            "battery": deque(maxlen=200),
            "timestamps": deque(maxlen=200)
        }
        
        # GPS数据
        self.gps_data = {}
        self.gps_valid = False
        self.gps_history = deque(maxlen=100)  # GPS历史轨迹
        
        # 检测结果
        self.yolo_detections = []
        self.pipeline_defects = []
        
        # 环境评估
        self.environment_assessment = {
            "level": "良好",
            "score": 75.0
        }
        
        # 系统健康
        self.system_health = {}
        
        # 管道信息
        self.pipeline_info = {
            "diameter": 0.0,      # 管道直径(mm)
            "material": "未知",   # 管道材质
            "distance": 0.0,      # 探测距离(m)
            "defect_count": 0,    # 缺陷数量
            "medium": "未知"      # 管道介质
        }
        
        # 警报状态
        self.alerts = {
            "voc_alert": False,
            "pressure_alert": False,
            "temperature_alert": False,
            "stability_alert": False,
            "battery_alert": False,
            "gps_alert": False,
            "defect_alert": False  # 管道缺陷警报
        }
        
        # 数据锁
        self.data_lock = threading.Lock()
    
    def update_sensor_data(self, data):
        """更新传感器数据 - 增强管道检测分析"""
        with self.data_lock:
            self.sensor_data = data
            
            # 更新历史数据
            if "temperature" in data:
                self.sensor_history["temperature"].append(data["temperature"])
            if "pressure" in data:
                self.sensor_history["pressure"].append(data["pressure"])
            if "air_quality" in data:
                self.sensor_history["air_quality"].append(data["air_quality"])
            if "battery" in data:
                self.sensor_history["battery"].append(data["battery"])
                
            self.sensor_history["timestamps"].append(time.time())
            
            # 处理GPS数据 - 兼容多种格式
            if self._extract_gps_from_sensor_data(data):
                # 如果传感器数据包含GPS信息，更新GPS状态
                pass
                
            # 更新环境评估
            if "environment_assessment" in data:
                self.environment_assessment = data["environment_assessment"]
            
            # 更新系统健康
            if "system_health" in data:
                self.system_health = data["system_health"]
            
            # 更新警报状态
            if "alerts" in data:
                self.alerts = data["alerts"]
                
            # 处理管道信息 (如果有)
            if "pipeline_info" in data:
                self.pipeline_info.update(data["pipeline_info"])
            
            # 处理管道缺陷数据 (如果有)
            if "pipeline_defects" in data:
                self.pipeline_defects = data["pipeline_defects"]
    
    def _extract_gps_from_sensor_data(self, data):
        """从传感器数据中提取GPS信息"""
        # 检查GPS有效性的多种方式
        gps_valid = False
        
        # 方式1: 直接检查gps_valid字段
        if "gps_valid" in data and data["gps_valid"]:
            gps_valid = True
        
        # 方式2: 检查经纬度是否存在且有效
        if ("latitude" in data and "longitude" in data and 
            data["latitude"] != 0 and data["longitude"] != 0):
            gps_valid = True
            
        if gps_valid:
            try:
                # 提取GPS数据
                lat = float(data.get("latitude", 0))
                lng = float(data.get("longitude", 0))
                
                # 验证坐标范围
                if abs(lat) <= 90 and abs(lng) <= 180 and (lat != 0 or lng != 0):
                    self.gps_valid = True
                    self.gps_data = {
                        "latitude": lat,
                        "longitude": lng,
                        "altitude": float(data.get("altitude", 0)),
                        "speed": float(data.get("gps_speed", data.get("speed", 0))),
                        "course": float(data.get("gps_course", data.get("course", 0))),
                        "satellites": int(data.get("gps_satellites", data.get("satellites", 0))),
                        "signal_strength": float(data.get("gps_signal_strength", data.get("signal_strength", 0))),
                        "accuracy": float(data.get("gps_accuracy", data.get("accuracy", 999))),
                        "timestamp": time.time(),
                        "valid": True
                    }
                    
                    # 添加到历史轨迹
                    self.gps_history.append({
                        "latitude": lat,
                        "longitude": lng,
                        "timestamp": time.time()
                    })
                    
                    return True
                    
            except (ValueError, TypeError) as e:
                print(f"GPS数据格式错误: {e}")
                
        return False
    
    def update_gps_data(self, gps_data):
        """专门更新GPS数据 - 增强版"""
        with self.data_lock:
            try:
                # 多种有效性检查方式
                valid_flags = [
                    gps_data.get("valid", False),
                    gps_data.get("is_valid", False),
                    gps_data.get("gps_valid", False)
                ]
                
                valid_flag = any(valid_flags)
                
                # 即使valid标志为False，也检查是否有有效坐标
                if not valid_flag and "latitude" in gps_data and "longitude" in gps_data:
                    try:
                        lat = float(gps_data["latitude"])
                        lng = float(gps_data["longitude"])
                        if abs(lat) <= 90 and abs(lng) <= 180 and (lat != 0 or lng != 0):
                            valid_flag = True
                    except (ValueError, TypeError):
                        pass
                
                if valid_flag:
                    # 确保坐标为浮点数
                    lat = float(gps_data.get("latitude", 0))
                    lng = float(gps_data.get("longitude", 0))
                    
                    # 验证坐标有效性
                    if abs(lat) > 90 or abs(lng) > 180:
                        print(f"GPS坐标超出有效范围: lat={lat}, lng={lng}")
                        return
                    
                    # 如果坐标为(0,0)，可能是无效数据
                    if lat == 0 and lng == 0:
                        print("GPS坐标为(0,0)，可能是无效数据")
                        return
                    
                    self.gps_valid = True
                    self.gps_data = {
                        "latitude": lat,
                        "longitude": lng,
                        "altitude": float(gps_data.get("altitude", 0)),
                        "speed": float(gps_data.get("speed", 0)),
                        "course": float(gps_data.get("course", 0)),
                        "satellites": int(gps_data.get("satellites", 0)),
                        "signal_strength": float(gps_data.get("signal_strength", 0)),
                        "accuracy": float(gps_data.get("accuracy", 999)),
                        "timestamp": time.time(),
                        "valid": True
                    }
                    
                    # 添加到历史轨迹
                    self.gps_history.append({
                        "latitude": lat,
                        "longitude": lng,
                        "timestamp": time.time()
                    })
                    
                else:
                    self.gps_valid = False
                    
            except Exception as e:
                print(f"处理GPS数据时出错: {e}")
                self.gps_valid = False
    
    def update_yolo_detections(self, detections):
        """更新YOLO检测结果"""
        with self.data_lock:
            self.yolo_detections = detections
            
            # 将YOLO检测结果转换为管道缺陷数据
            if detections:
                # 更新管道缺陷
                pipeline_defects = []
                
                # 遍历检测结果并转换格式
                for det in detections:
                    # 修正：确保名称是有效的字符串
                    if 'name' not in det or not isinstance(det['name'], str):
                        det['name'] = f"类别{det.get('class', 0)}"
                    
                    defect = {
                        'type': det.get('name', 'unknown'),
                        'severity': det.get('severity', '正常'),
                        'position': det.get('position', '未知'),
                        'size': det.get('size', '未知'),
                        'confidence': det.get('confidence', 0.0),
                        'location': [det['box'][0], det['box'][1], 
                                    det['box'][2]-det['box'][0], 
                                    det['box'][3]-det['box'][1]]
                    }
                    pipeline_defects.append(defect)
                
                # 更新缺陷列表
                self.pipeline_defects = pipeline_defects
                
                # 更新缺陷计数
                self.pipeline_info["defect_count"] = len(pipeline_defects)
                
                # 设置缺陷警报
                if any(d.get('severity') == '严重' for d in pipeline_defects):
                    self.alerts["defect_alert"] = True
                else:
                    self.alerts["defect_alert"] = False
    
    def update_pipeline_defects(self, defects):
        """更新管道缺陷信息"""
        with self.data_lock:
            self.pipeline_defects = defects
            
            # 设置缺陷警报
            if defects and any(d.get('severity') == '严重' for d in defects):
                self.alerts["defect_alert"] = True
            else:
                self.alerts["defect_alert"] = False
                
            # 更新缺陷计数
            self.pipeline_info["defect_count"] = len(defects) if defects else 0
    
    def get_sensor_data(self):
        """获取最新传感器数据"""
        with self.data_lock:
            return self.sensor_data.copy()
    
    def get_gps_data(self):
        """获取GPS数据"""
        with self.data_lock:
            return self.gps_data.copy() if self.gps_data else {}
    
    def get_gps_valid(self):
        """获取GPS有效状态"""
        with self.data_lock:
            return self.gps_valid
    
    def get_gps_track(self, count=50):
        """获取GPS轨迹历史"""
        with self.data_lock:
            return list(self.gps_history)[-count:] if self.gps_history else []
    
    def get_sensor_history(self):
        """获取传感器历史数据"""
        with self.data_lock:
            return {k: list(v) for k, v in self.sensor_history.items()}
    
    def get_yolo_detections(self):
        """获取YOLO检测结果"""
        with self.data_lock:
            return self.yolo_detections.copy() if self.yolo_detections else []
            
    def get_pipeline_defects(self):
        """获取管道缺陷信息"""
        with self.data_lock:
            return self.pipeline_defects.copy() if self.pipeline_defects else []
            
    def get_pipeline_info(self):
        """获取管道信息"""
        with self.data_lock:
            return self.pipeline_info.copy()
    
    def get_environment_assessment(self):
        """获取环境评估"""
        with self.data_lock:
            return self.environment_assessment.copy()
    
    def get_system_health(self):
        """获取系统健康"""
        with self.data_lock:
            return self.system_health.copy()
    
    def get_alerts(self):
        """获取警报状态"""
        with self.data_lock:
            return self.alerts.copy()
    
    def analyze_situation(self):
        """综合分析当前情况 - 针对管道检测场景增强"""
        with self.data_lock:
            data = self.sensor_data
            detections = self.yolo_detections
            defects = self.pipeline_defects
            alerts = self.alerts
            
            analysis = {
                "timestamp": time.time(),
                "summary": "管道状态正常",
                "risk_level": "低",
                "recommendations": ["继续正常巡检"],
                "critical_findings": []
            }
            
            # 检测缺陷风险
            has_defects = bool(defects)
            has_severe_defects = any(d.get('severity') == '严重' for d in defects) if defects else False
            
            # 检测是否有人
            has_person = any(d.get("name") == "person" for d in detections) if detections else False
            
            # 检测是否有警报
            has_alerts = any(alerts.values()) if alerts else False
            
            # GPS状态分析
            gps_quality = "良好" if self.gps_valid else "无效"
            if self.gps_valid and self.gps_data:
                signal_strength = self.gps_data.get("signal_strength", 0)
                if signal_strength < 70:
                    gps_quality = "较差"
                elif signal_strength < 85:
                    gps_quality = "一般"
            
            # 风险等级评估
            risk_factors = []
            
            # 1. 管道缺陷风险
            if has_defects:
                if has_severe_defects:
                    risk_factors.append("发现严重管道缺陷")
                    analysis["critical_findings"].append("检测到严重的管道缺陷，需要立即处理")
                else:
                    risk_factors.append("发现一般管道缺陷")
                    
            # 2. 人员检测风险
            if has_person:
                risk_factors.append("检测到人员")
                analysis["critical_findings"].append("检测到管道内或附近有人员活动")
                
            # 3. 环境风险
            env_level = self.environment_assessment.get("level", "良好")
            if env_level in ["较差", "危险"]:
                risk_factors.append(f"环境质量{env_level}")
                analysis["critical_findings"].append(f"环境评估为{env_level}级别")
            
            # 4. 各种警报风险
            if alerts.get("voc_alert", False):
                risk_factors.append("VOC指数异常")
                analysis["critical_findings"].append("空气质量异常，可能存在有害气体")
                
            if alerts.get("pressure_alert", False):
                risk_factors.append("气压快速变化")
                analysis["critical_findings"].append("气压快速变化，可能预示管道压力异常")
                
            if alerts.get("temperature_alert", False):
                risk_factors.append("温度异常")
                analysis["critical_findings"].append("温度异常，可能存在管道过热点")
                
            if alerts.get("stability_alert", False):
                risk_factors.append("稳定性差")
                analysis["critical_findings"].append("机器人稳定性差，可能影响检测精度")
                
            if alerts.get("battery_alert", False):
                risk_factors.append("电池电量低")
                
            if alerts.get("gps_alert", False):
                risk_factors.append("GPS信号异常")
                analysis["critical_findings"].append("GPS信号质量差，可能影响定位精度")
            
            # 综合风险评估
            if has_severe_defects or len(risk_factors) >= 3:
                analysis["risk_level"] = "高"
                analysis["summary"] = "发现严重问题，建议立即处理"
                analysis["recommendations"] = ["立即结束检测任务", "向监控中心报告情况", "安排维修队伍处理"]
            elif has_defects or len(risk_factors) >= 1:
                analysis["risk_level"] = "中"
                analysis["summary"] = "存在潜在问题"
                analysis["recommendations"] = ["谨慎操作", "持续监测", "记录问题位置以便后续处理"]
            else:
                analysis["risk_level"] = "低"
                analysis["summary"] = "管道状态正常，无明显风险"
                analysis["recommendations"] = ["继续正常巡检"]
            
            # 生成详细分析结果
            analysis["risk_factors"] = risk_factors
            analysis["gps_quality"] = gps_quality
            analysis["detailed_data"] = {
                "temperature": data.get("temperature", 0),
                "pressure": data.get("pressure", 0),
                "air_quality": data.get("air_quality", 0),
                "stability_index": data.get("stability_index", 0),
                "battery": data.get("battery", 0),
                "defect_count": len(defects) if defects else 0,
                "pipeline_diameter": self.pipeline_info.get("diameter", 0),
                "pipeline_material": self.pipeline_info.get("material", "未知"),
                "inspection_distance": self.pipeline_info.get("distance", 0),
                "human_detections": sum(1 for d in detections if d.get("name") == "person") if detections else 0,
                "gps_satellites": self.gps_data.get("satellites", 0) if self.gps_valid else 0,
                "gps_signal_strength": self.gps_data.get("signal_strength", 0) if self.gps_valid else 0
            }
            
            return analysis



#====================== 自定义UI组件 ======================

class HexagonButton(QPushButton):
    """六边形按钮，具有高科技外观"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(120, 80)
        self.setMaximumHeight(80)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    
    def paintEvent(self, event):
        # 创建绘制对象
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 获取尺寸
        w = self.width()
        h = self.height()
        
        # 设置画笔和填充颜色
        if self.isChecked():
            bg_color = QColor("#00E0E0")
            text_color = QColor("#000000")
            border_color = QColor("#00E0E0")
        else:
            bg_color = QColor(30, 35, 50, 180)
            text_color = QColor("#FFFFFF")
            border_color = QColor("#00CCCC")
        
        # 鼠标悬停效果
        if self.underMouse() and not self.isChecked():
            bg_color = QColor(40, 45, 60, 200)
        
        # 绘制六边形
        path = QPainterPath()
        side = min(w, h) / 4
        
        points = [
            QPoint(side, 0),           # 上
            QPoint(w - side, 0),       # 上右
            QPoint(w, h/2),            # 右
            QPoint(w - side, h),       # 下右
            QPoint(side, h),           # 下
            QPoint(0, h/2)             # 左
        ]
        
        path.moveTo(points[0])
        for i in range(1, 6):
            path.lineTo(points[i])
        path.closeSubpath()
        
        # 填充和描边
        painter.setPen(QPen(border_color, 2))
        painter.setBrush(QBrush(bg_color))
        painter.drawPath(path)
        
        # 添加发光效果
        if self.isChecked():
            glow_path = QPainterPath(path)
            painter.setPen(QPen(QColor(0, 255, 255, 100), 6))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(glow_path)
        
        # 绘制文本 - 调整字体大小
        font = painter.font()
        font.setPointSize(32)  # 在这里修改字体大小，默认是12
        font.setBold(True)     # 设置为粗体
        painter.setFont(font)
        painter.setPen(text_color)
        painter.drawText(path.boundingRect(), Qt.AlignCenter, self.text())

class TechDirectionButton(QPushButton):
    """科技风格的方向按钮，带角度边框和发光效果"""
    def __init__(self, direction, parent=None):
        super().__init__("", parent)
        self.direction = direction
        self.setFixedSize(70, 70)
        self.setCheckable(True)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 设置基本样式
        if self.isChecked():
            pen_color = QColor("#00E0E0")
            glow_color = QColor(0, 224, 224, 100)
            bg_color = QColor(0, 224, 224, 40)
        else:
            pen_color = QColor("#00AACC")
            glow_color = QColor(0, 170, 204, 40)
            bg_color = QColor(0, 0, 0, 0)
            
            # 鼠标悬停效果
            if self.underMouse():
                pen_color = QColor("#00CCDD")
                bg_color = QColor(0, 204, 221, 20)
        
        # 绘制背景
        path = self._create_button_path()
        painter.setPen(QPen(pen_color, 2))
        painter.setBrush(QBrush(bg_color))
        painter.drawPath(path)
        
        # 绘制箭头
        arrow_path = self._create_arrow_path()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(pen_color))
        painter.drawPath(arrow_path)
        
        # 发光效果
        if self.isChecked():
            painter.setPen(QPen(glow_color, 6))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
    
    def _create_button_path(self):
        """创建按钮外形路径 - 带尖角的六边形"""
        path = QPainterPath()
        w, h = self.width(), self.height()
        
        # 按钮六边形设计
        points = [
            QPoint(w/2, 10),       # 上
            QPoint(w-10, h/3),     # 右上
            QPoint(w-10, 2*h/3),   # 右下
            QPoint(w/2, h-10),     # 下
            QPoint(10, 2*h/3),     # 左下
            QPoint(10, h/3)        # 左上
        ]
        
        path.moveTo(points[0])
        for i in range(1, 6):
            path.lineTo(points[i])
        path.closeSubpath()
        
        return path
    
    def _create_arrow_path(self):
        """创建方向箭头路径"""
        path = QPainterPath()
        w, h = self.width(), self.height()
        center_x, center_y = w//2, h//2
        
        if self.direction == "前进":
            # 上箭头
            points = [
                QPoint(center_x, center_y-15),
                QPoint(center_x-12, center_y+5),
                QPoint(center_x+12, center_y+5)
            ]
        elif self.direction == "后退":
            # 下箭头
            points = [
                QPoint(center_x, center_y+15),
                QPoint(center_x-12, center_y-5),
                QPoint(center_x+12, center_y-5)
            ]
        elif self.direction == "左转":
            # 左箭头
            points = [
                QPoint(center_x-15, center_y),
                QPoint(center_x+5, center_y-12),
                QPoint(center_x+5, center_y+12)
            ]
        elif self.direction == "右转":
            # 右箭头
            points = [
                QPoint(center_x+15, center_y),
                QPoint(center_x-5, center_y-12),
                QPoint(center_x-5, center_y+12)
            ]
        elif self.direction == "复位":
            # 循环箭头
            path.arcMoveTo(center_x-12, center_y-12, 24, 24, 45)
            path.arcTo(center_x-12, center_y-12, 24, 24, 45, 270)
            
            # 箭头尖
            arrow_head = [
                QPoint(center_x, center_y-15),
                QPoint(center_x-5, center_y-5),
                QPoint(center_x+5, center_y-5)
            ]
            
            path.moveTo(arrow_head[0])
            path.lineTo(arrow_head[1])
            path.lineTo(arrow_head[2])
            path.closeSubpath()
            
            return path
        
        # 常规箭头路径
        path.moveTo(points[0])
        path.lineTo(points[1])
        path.lineTo(points[2])
        path.closeSubpath()
        
        return path

class PipelineVideoWidget(QLabel):
    """管道检测视频流显示控件"""
    frame_ready = Signal(np.ndarray)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignCenter)
        
        # 初始化资源管理器
        self.resource_manager = ResourceManager()
        self.video_processor = self.resource_manager.get_video_processor()
        
        # 视频处理配置
        self.fps_limit = 30
        self.last_frame_time = 0
        
        # 样式设置 - 管道风格
        self.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #101820, stop:1 #0A1015);
                border: 3px solid #00CCCC;
                border-radius: 5px;
            }
        """)
        
        # 显示默认图像
        self.show_default_image()
        
        # YOLO检测器 - 使用增强的检测器
        self.yolo_detector = None
        self.detection_enabled = False
        
        # 性能监控
        self.fps_counter = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)
        self.current_fps = 0
        
        # 当前显示的帧
        self.current_frame = None
        self.current_processed_frame = None
        
        # 管道缺陷检测结果
        self.defect_data = []
        
        # 辅助信息显示开关
        self.show_info = True
        self.show_grid = True
        self.show_crosshair = True
        
    def show_default_image(self):
        """显示默认图像"""
        pixmap = QPixmap(640, 480)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 渐变背景
        gradient = QLinearGradient(0, 0, pixmap.width(), pixmap.height())
        gradient.setColorAt(0, QColor(16, 24, 32))
        gradient.setColorAt(1, QColor(10, 16, 21))
        painter.fillRect(pixmap.rect(), gradient)
        
        # 管道轮廓特效
        painter.setPen(QPen(QColor("#00DDDD"), 2, Qt.DashLine))
        painter.drawEllipse(pixmap.width()/2 - 200, pixmap.height()/2 - 100, 400, 200)
        
        # 添加环形标记
        painter.setPen(QPen(QColor("#00CCCC"), 2))
        painter.drawEllipse(pixmap.width()/2 - 180, pixmap.height()/2 - 90, 360, 180)
        painter.setPen(QPen(QColor("#009999"), 1))
        painter.drawEllipse(pixmap.width()/2 - 160, pixmap.height()/2 - 80, 320, 160)
        
        # 十字准星
        painter.setPen(QPen(QColor("#00DDDD"), 1, Qt.DotLine))
        painter.drawLine(pixmap.width()/2, 0, pixmap.width()/2, pixmap.height())
        painter.drawLine(0, pixmap.height()/2, pixmap.width(), pixmap.height()/2)
        
        # 绘制机器人图标
        robot_path = QPainterPath()
        robot_path.addRoundedRect(QRectF(pixmap.width()/2-30, pixmap.height()/2-15, 60, 30), 10, 10)
        
        painter.setPen(QPen(QColor("#00FFFF"), 2))
        painter.setBrush(QBrush(QColor(0, 255, 255, 80)))
        painter.drawPath(robot_path)
        
        # 文字
        painter.setPen(QColor("#00DDDD"))
        painter.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, 
                        "🎥 等待视频流...\n\n正在连接管道检测机器人摄像头")
        
        # 边框装饰
        painter.setPen(QPen(QColor("#00CCCC"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(10, 10, pixmap.width()-20, pixmap.height()-20)
        
        # 角落装饰
        corner_size = 40
        painter.drawLine(10, 10, 10+corner_size, 10)
        painter.drawLine(10, 10, 10, 10+corner_size)
        
        painter.drawLine(pixmap.width()-10, 10, pixmap.width()-10-corner_size, 10)
        painter.drawLine(pixmap.width()-10, 10, pixmap.width()-10, 10+corner_size)
        
        painter.drawLine(10, pixmap.height()-10, 10+corner_size, pixmap.height()-10)
        painter.drawLine(10, pixmap.height()-10, 10, pixmap.height()-10-corner_size)
        
        painter.drawLine(pixmap.width()-10, pixmap.height()-10, pixmap.width()-10-corner_size, pixmap.height()-10)
        painter.drawLine(pixmap.width()-10, pixmap.height()-10, pixmap.width()-10, pixmap.height()-10-corner_size)
        
        # 添加测量标记
        painter.setPen(QPen(QColor("#00BBBB"), 1, Qt.DotLine))
        for i in range(0, pixmap.width(), 80):
            painter.drawLine(i, 0, i, 10)
            painter.drawLine(i, pixmap.height()-10, i, pixmap.height())
        
        for i in range(0, pixmap.height(), 80):
            painter.drawLine(0, i, 10, i)
            painter.drawLine(pixmap.width()-10, i, pixmap.width(), i)
            
        painter.end()
        self.setPixmap(pixmap)
        self.current_frame = None
        self.current_processed_frame = None
    
    def update_frame(self, cv_frame):
        """更新视频帧"""
        current_time = time.time()
        
        # 帧率限制
        if current_time - self.last_frame_time < 1.0 / self.fps_limit:
            return
        
        self.last_frame_time = current_time
        
        try:
            # 保存当前帧
            self.current_frame = cv_frame.copy()
            
            # 处理帧
            processed_frame = self.video_processor.process_frame(cv_frame.copy())
            
            # YOLO检测 - 只在启用时执行
            if self.detection_enabled and self.yolo_detector:
                processed_frame, detections = self.yolo_detector.detect(processed_frame, draw=True)
                # 发送检测结果信号或更新数据融合系统
                self.frame_ready.emit(processed_frame)
            else:
                # 检测未启用时，不进行对象检测
                                detections = []
            
            # 添加管道检测辅助信息
            if self.show_info or self.show_grid or self.show_crosshair:
                processed_frame = self._add_pipeline_inspection_overlay(processed_frame)
            
            # 保存处理后的帧
            self.current_processed_frame = processed_frame
            
            # 转换为Qt格式
            qt_image = self.video_processor.convert_to_qt(processed_frame)
            if qt_image:
                pixmap = QPixmap.fromImage(qt_image)
                self.setPixmap(pixmap)
                
                # 发射信号
                self.frame_ready.emit(cv_frame)
                
                # 更新FPS计数
                self.fps_counter += 1
                
        except Exception as e:
            print(f"视频帧处理错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_pipeline_inspection_overlay(self, frame):
        """添加管道检测辅助信息覆盖层"""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # 添加十字准星
        if self.show_crosshair:
            cv2.line(overlay, (w//2, 0), (w//2, h), (0, 224, 224, 128), 1, cv2.LINE_AA)
            cv2.line(overlay, (0, h//2), (w, h//2), (0, 224, 224, 128), 1, cv2.LINE_AA)
            
            # 添加中心圆
            cv2.circle(overlay, (w//2, h//2), 20, (0, 224, 224, 128), 1, cv2.LINE_AA)
            cv2.circle(overlay, (w//2, h//2), 5, (0, 224, 224, 128), -1, cv2.LINE_AA)
        
        # 添加测量网格
        if self.show_grid:
            # 纵向网格
            for i in range(0, w, 80):
                cv2.line(overlay, (i, 0), (i, h), (0, 204, 204, 64), 1, cv2.LINE_AA)
                
            # 横向网格
            for i in range(0, h, 80):
                cv2.line(overlay, (0, i), (w, i), (0, 204, 204, 64), 1, cv2.LINE_AA)
        
        # 添加管道缺陷标记 - 只有在启用检测时才添加
        if self.detection_enabled and self.defect_data:
            for defect in self.defect_data:
                try:
                    if 'location' in defect and len(defect['location']) == 4:
                        x, y, width, height = defect['location']
                        severity = defect.get('severity', '一般')
                        defect_type = defect.get('type', '未知')
                        
                        # 根据严重程度选择颜色
                        if severity == '严重':
                            color = (85, 85, 255)  # 红色
                        elif severity == '中等':
                            color = (85, 149, 255)  # 橙色
                        else:
                            color = (0, 204, 204)  # 蓝绿色
                        
                        # 绘制矩形框
                        cv2.rectangle(overlay, (x, y), (x+width, y+height), color, 2, cv2.LINE_AA)
                        
                        # 绘制标签背景
                        label = f"{defect_type} ({severity})"
                        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(overlay, (x, y-label_h-5), (x+label_w+5, y), color, -1)
                        
                        # 绘制标签文字
                        cv2.putText(overlay, label, (x+3, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"绘制缺陷标记错误: {e}")
        
        # 添加信息面板
        if self.show_info:
            info_h = 80
            info_w = 160
            
            # 创建半透明背景
            cv2.rectangle(overlay, (10, 10), (10+info_w, 10+info_h), (0, 0, 0, 160), -1)
            cv2.rectangle(overlay, (10, 10), (10+info_w, 10+info_h), (0, 224, 224), 1, cv2.LINE_AA)
            
            # FPS信息
            cv2.putText(overlay, f"FPS: {self.current_fps}", (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            # 分辨率
            cv2.putText(overlay, f"分辨率: {w}x{h}", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            # 当前模式
            if self.detection_enabled:
                mode_text = "智能检测"
            else:
                mode_text = "标准检测"
                
            cv2.putText(overlay, f"模式: {mode_text}", (15, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            # 右上角添加时间戳
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            (time_w, time_h), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(overlay, time_str, (w-time_w-15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        return overlay
    
    def update_fps(self):
        """更新FPS显示"""
        self.current_fps = self.fps_counter
        self.fps_counter = 0
        
    def toggle_info(self):
        """切换信息显示"""
        self.show_info = not self.show_info
        
    def toggle_grid(self):
        """切换网格显示"""
        self.show_grid = not self.show_grid
        
    def toggle_crosshair(self):
        """切换十字准星显示"""
        self.show_crosshair = not self.show_crosshair
    
    def update_defect_data(self, defect_data):
        """更新缺陷数据"""
        self.defect_data = defect_data
    
    def enable_yolo_detection(self, enable=True):
        """启用/禁用YOLO检测 - 使用yolo11模型"""
        if not YOLO_AVAILABLE:
            print("YOLO库不可用，无法启用智能检测")
            return False
            
        if enable:
            # 如果要启用检测但没有检测器，则创建一个
            if self.yolo_detector is None:
                try:
                    # 检查模型文件是否存在
                    if not os.path.exists(YOLO_MODEL_PATH):
                        print(f"错误：YOLO模型文件不存在: {YOLO_MODEL_PATH}")
                        return False
                        
                    print(f"正在加载YOLO模型: {YOLO_MODEL_PATH}")
                    self.yolo_detector = YOLODetector(YOLO_MODEL_PATH)
                    if self.yolo_detector.initialize():
                        self.detection_enabled = True
                        print("YOLO检测器初始化成功")
                        return True
                    else:
                        print("YOLO检测器初始化失败")
                        self.yolo_detector = None
                        return False
                except Exception as e:
                    print(f"YOLO初始化失败: {e}")
                    import traceback
                    traceback.print_exc()
                    self.yolo_detector = None
                    return False
            else:
                # 已有检测器，只需启用
                self.detection_enabled = True
                return True
        else:
            # 禁用检测
            self.detection_enabled = False
            return True
    
    def get_current_frame(self):
        """获取当前显示的帧"""
        return self.current_frame
        
    def get_current_processed_frame(self):
        """获取当前处理后的帧"""
        return self.current_processed_frame

class PipelineDefectWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 标题 - 单独设置字体大小
        title_layout = QHBoxLayout()
        
        # icon_label = QLabel("🔍")
        # icon_label.setStyleSheet("""
        #     font-size: 26pt;  /* 图标字体大小 */
        #     color: #00CCCC;
        # """)
        
        # title_label = QLabel("管道缺陷检测")
        # title_label.setStyleSheet("""
        #     color: #FFFFFF;
        #     font-size: 22pt;  /* 标题字体大小 */
        #     font-weight: bold;
        # """)
        
        # title_layout.addWidget(icon_label)
        # title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # 状态显示 - 单独设置字体大小
        self.status_label = QLabel("等待检测数据...")
        self.status_label.setStyleSheet("""
            color: #AAAAAA;
            font-size: 14pt;  /* 状态显示字体大小 */
            padding: 5px;
        """)
        
        # 检测结果表格 - 单独设置字体大小
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["缺陷类型", "严重程度", "位置", "尺寸"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setStyleSheet("""
            QTableWidget {
                background: rgba(16, 24, 32, 120);
                border: none;
                border-radius: 5px;
                gridline-color: rgba(0, 204, 204, 50);
                color: #FFFFFF;
                font-size: 14pt;  /* 表格内容字体大小 */
            }
            QHeaderView::section {
                background: rgba(0, 204, 204, 150);
                color: #FFFFFF;
                font-weight: bold;
                font-size: 16pt;  /* 表格标题字体大小 */
                border: none;
                padding: 5px;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid rgba(0, 204, 204, 30);
            }
            QTableWidget::item:selected {
                background: rgba(0, 204, 204, 100);
            }
        """)
        
        # 管道信息显示 - 单独设置字体大小
        self.pipe_info_label = QLabel("管道信息: 等待数据...")
        self.pipe_info_label.setStyleSheet("""
            color: #FFFFFF;
            font-size: 14pt;  /* 管道信息字体大小 */
            padding: 5px;
            background: rgba(16, 24, 32, 120);
            border-radius: 5px;
        """)
        self.pipe_info_label.setAlignment(Qt.AlignCenter)
        
        # YOLO模型信息 - 单独设置字体大小
        self.model_info_label = QLabel("模型: yolo11.pt")
        self.model_info_label.setStyleSheet("""
            color: #00CCCC;
            font-size: 14pt;  /* 模型信息字体大小 */
            padding: 5px;
            background: rgba(16, 24, 32, 120);
            border-radius: 5px;
        """)
        self.model_info_label.setAlignment(Qt.AlignCenter)
        
        # 添加到布局
        layout.addLayout(title_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.results_table, 1)
        layout.addWidget(self.pipe_info_label)
        layout.addWidget(self.model_info_label)
        
        # 整体样式
        self.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(16, 24, 32, 220),
                stop:1 rgba(10, 16, 21, 220));
            border: 2px solid #00CCCC;
            border-radius: 5px;
        """)
    
    def update_defects(self, defects):
        """更新缺陷检测结果"""
        # 清除现有行
        self.results_table.setRowCount(0)
        
        if not defects:
            self.status_label.setText("未检测到管道缺陷")
            return
            
        # 更新状态
        self.status_label.setText(f"检测到 {len(defects)} 个缺陷点")
        
        # 添加检测结果
        for i, defect in enumerate(defects):
            self.results_table.insertRow(i)
            
            # 缺陷类型
            defect_type = defect.get('type', '未知')
            type_item = QTableWidgetItem(defect_type)
            
            # 严重程度
            severity = defect.get('severity', '一般')
            severity_item = QTableWidgetItem(severity)
            
            # 位置
            position = defect.get('position', '未知')
            pos_item = QTableWidgetItem(position)
            
            # 尺寸
            size = defect.get('size', '未知')
            size_item = QTableWidgetItem(size)
            
            # 根据严重程度设置颜色
            if severity == '严重':
                color = QColor("#FF5555")
            elif severity == '中等':
                color = QColor("#FF9500")
            elif severity == '轻微':
                color = QColor("#FFCC00")
            else:
                color = QColor("#00CCCC")
                
            # 设置表格项
            for col, item in enumerate([type_item, severity_item, pos_item, size_item]):
                item.setForeground(color)
                self.results_table.setItem(i, col, item)
    
    def update_pipeline_info(self, pipe_info):
        """更新管道信息"""
        if not pipe_info:
            self.pipe_info_label.setText("管道信息: 数据不可用")
            return
            
        # 获取信息
        diameter = pipe_info.get("diameter", 0)
        material = pipe_info.get("material", "未知")
        distance = pipe_info.get("distance", 0)
        defect_count = pipe_info.get("defect_count", 0)
        medium = pipe_info.get("medium", "未知")
        
        # 更新显示
        self.pipe_info_label.setText(
            f"管道信息: 直径: {diameter}mm | "
            f"材质: {material} | "
            f"介质: {medium} | "
            f"检测距离: {distance:.1f}m | "
            f"缺陷总数: {defect_count}个"
        )
    
    def update_model_info(self, model_path, classes=None):
        """更新模型信息"""
        model_name = os.path.basename(model_path)
        if classes:
            class_str = ", ".join(classes[:5])
            if len(classes) > 5:
                class_str += "..."
            self.model_info_label.setText(f"模型: {model_name} | 类别: {class_str}")
        else:
            self.model_info_label.setText(f"模型: {model_name}")

class MQTTThread(QThread):
    """MQTT通信线程 - 管道检测机器人版本"""
    sensor_data_signal = Signal(dict)
    gps_data_signal = Signal(dict)
    camera_frame_signal = Signal(np.ndarray)
    camera_normal_frame_signal = Signal(np.ndarray)
    connection_signal = Signal(bool)
    alert_signal = Signal(dict)
    log_signal = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_connected = False
        self.client = None
        self.should_stop = False
        
    def run(self):
        try:
            # 使用随机后缀避免客户端ID冲突
            client_id = f"{MQTT_CLIENT_ID}_{int(time.time())}"
            self.log_signal.emit(f"尝试连接MQTT服务器: {MQTT_BROKER}:{MQTT_PORT}")
            self.log_signal.emit(f"使用客户端ID: {client_id}")
            
            # 创建MQTT客户端
            self.client = mqtt.Client(client_id=client_id)
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect
            
            # 设置认证
            if MQTT_USERNAME and MQTT_PASSWORD:
                self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
                self.log_signal.emit(f"已设置MQTT认证: 用户={MQTT_USERNAME}")
            
            # 连接服务器
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            
            # 启动循环
            self.log_signal.emit("MQTT客户端启动循环...")
            self.client.loop_forever()
            
        except Exception as e:
            self.log_signal.emit(f"MQTT连接错误: {e}")
            self.connection_signal.emit(False)
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.is_connected = True
            self.connection_signal.emit(True)
            self.log_signal.emit("✅ MQTT连接成功！")
            
            # 订阅所有需要的主题
            topics = [
                (MQTT_TOPIC_SUBSCRIBE, 1),
                (MQTT_TOPIC_STATUS, 1),
                (MQTT_TOPIC_CAMERA, 0),
                (MQTT_TOPIC_CAMERA_NORMAL, 0),
                (MQTT_TOPIC_GPS, 1),
                (MQTT_TOPIC_ALERTS, 1)
            ]
            
            for topic, qos in topics:
                client.subscribe(topic, qos)
                self.log_signal.emit(f"已订阅主题: {topic}")
        else:
            self.is_connected = False
            error_msgs = {
                1: "协议版本错误",
                2: "客户端ID无效",
                3: "服务器不可用",
                4: "用户名或密码错误",
                5: "未授权"
            }
            error_msg = error_msgs.get(rc, f"未知错误(码:{rc})")
            self.log_signal.emit(f"❌ MQTT连接失败: {error_msg}")
            self.connection_signal.emit(False)
    
    def on_disconnect(self, client, userdata, rc):
        self.is_connected = False
        self.connection_signal.emit(False)
        self.log_signal.emit(f"与MQTT代理断开连接，返回码: {rc}")
    
    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload_size = len(msg.payload)
            
            # 减少图像主题的日志输出
            if topic not in [MQTT_TOPIC_CAMERA, MQTT_TOPIC_CAMERA_NORMAL]:
                self.log_signal.emit(f"收到消息: {topic} ({payload_size}字节)")
            
            # 解析JSON数据
            try:
                data = json.loads(msg.payload.decode('utf-8'))
            except Exception as e:
                self.log_signal.emit(f"消息解析错误: {e}")
                return
            
            # 根据主题分发消息
            if topic == MQTT_TOPIC_STATUS:
                # 处理状态数据（可能包含GPS）
                self.sensor_data_signal.emit(data)
                
                # 如果状态数据包含GPS信息，也发送GPS数据信号
                if self._extract_gps_from_status(data):
                    pass  # GPS数据已在extract函数中处理
                
            elif topic == MQTT_TOPIC_GPS:
                # 专门处理GPS数据
                self._handle_gps_message(data)
                
            elif topic == MQTT_TOPIC_CAMERA or topic == MQTT_TOPIC_CAMERA_NORMAL:
                self._process_image_data(data, topic)
                
            elif topic == MQTT_TOPIC_ALERTS:
                self.alert_signal.emit(data)
                
        except Exception as e:
            self.log_signal.emit(f"消息处理错误: {e}")
    
    def _extract_gps_from_status(self, data):
        """从状态数据中提取GPS信息"""
        try:
            # 检查是否包含GPS数据
            if ("latitude" in data and "longitude" in data and 
                data.get("gps_valid", False)):
                
                # 验证GPS数据有效性
                lat = float(data["latitude"])
                lng = float(data["longitude"])
                
                if abs(lat) <= 90 and abs(lng) <= 180 and (lat != 0 or lng != 0):
                    # 构建GPS数据包
                    gps_data = {
                        "latitude": lat,
                        "longitude": lng,
                        "altitude": float(data.get("altitude", 0)),
                        "speed": float(data.get("gps_speed", data.get("speed", 0))),
                        "course": float(data.get("gps_course", data.get("course", 0))),
                        "satellites": int(data.get("gps_satellites", data.get("satellites", 0))),
                        "signal_strength": float(data.get("gps_signal_strength", data.get("signal_strength", 0))),
                        "accuracy": float(data.get("gps_accuracy", data.get("accuracy", 999))),
                        "timestamp": time.time(),
                        "valid": True
                    }
                    
                    # 发射GPS数据信号
                    self.gps_data_signal.emit(gps_data)
                    self.log_signal.emit(f"从状态数据提取GPS: lat={lat:.6f}, lng={lng:.6f}")
                    return True
                    
        except (ValueError, TypeError, KeyError) as e:
            self.log_signal.emit(f"提取状态数据中的GPS信息失败: {e}")
            
        return False
    
    def _handle_gps_message(self, data):
        """处理专门的GPS消息"""
        try:
            # 验证GPS数据格式
            if "latitude" not in data or "longitude" not in data:
                self.log_signal.emit("GPS数据格式错误: 缺少经纬度字段")
                return
            
            # 类型转换和验证
            lat = float(data["latitude"])
            lng = float(data["longitude"])
            
            # 验证坐标有效性
            if abs(lat) > 90 or abs(lng) > 180:
                self.log_signal.emit(f"GPS坐标超出有效范围: lat={lat}, lng={lng}")
                return
                
            # 排除无效坐标
            if lat == 0 and lng == 0:
                self.log_signal.emit("GPS坐标为(0,0)，可能是无效数据")
                return
            
            # 确保数据包含valid字段
            if "valid" not in data:
                data["valid"] = True
                
            # 发射GPS数据信号
            self.gps_data_signal.emit(data)
            self.log_signal.emit(f"收到专门GPS数据: lat={lat:.6f}, lng={lng:.6f}")
            
        except (ValueError, TypeError) as e:
            self.log_signal.emit(f"处理GPS消息时出错: {e}")
    
    def _process_image_data(self, data, topic):
        """处理图像数据"""
        try:
            if "data" not in data:
                return
            
            # 解码base64图像
            img_bytes = base64.b64decode(data["data"])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            cv_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_frame is not None and cv_frame.size > 0:
                # 发射相应信号
                if topic == MQTT_TOPIC_CAMERA_NORMAL:
                    self.camera_normal_frame_signal.emit(cv_frame)
                elif topic == MQTT_TOPIC_CAMERA:
                    self.camera_frame_signal.emit(cv_frame)
            else:
                self.log_signal.emit(f"{topic}图像解码失败")
                
        except Exception as e:
            self.log_signal.emit(f"{topic}图像处理错误: {e}")
    
    def publish_command(self, command):
        """发布控制命令"""
        if not self.is_connected or not self.client:
            self.log_signal.emit("MQTT未连接，无法发送命令")
            return False
        
        try:
            # 添加时间戳
            command.update({
                "timestamp": time.time(),
                "user": CURRENT_USER
            })
            
            payload = json.dumps(command)
            self.client.publish(MQTT_TOPIC_PUBLISH, payload)
            self.log_signal.emit(f"已发布命令: {command}")
            return True
            
        except Exception as e:
            self.log_signal.emit(f"发布命令错误: {e}")
            return False
    
    def stop_thread(self):
        """停止线程"""
        self.should_stop = True
        if self.client:
            try:
                self.client.disconnect()
            except:
                pass

class PipelineRobotDashboard(QMainWindow):
    """管道检测机器人控制面板"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"管道检测机器人控制系统 v2.0 - {CURRENT_USER}")
        self.setMinimumSize(1280, 960)  # 调整主窗口大小
        
        # 初始化资源管理器
        self.resource_manager = ResourceManager()
        
        # 多模态数据融合系统
        self.data_fusion = MultiModalDataFusion()
        
        # 当前状态
        self.current_mode = None
        self.current_direction = None
        
        # YOLO检测
        self.yolo_detector = None
        self.mqtt_detection_enabled = False
        
        # GPS配置
        self.gps_real_time_enabled = True
        self.last_gps_update = 0
        self.last_map_update_time = 0
        self.gps_update_count = 0  # GPS更新计数
        
        # MQTT连接状态
        self.mqtt_connected = False
        self.mqtt_connection_attempts = 0
        
        # 调试日志窗口
        self.debug_log = QTextEdit()
        self.debug_log.setReadOnly(True)
        self.debug_log.setFixedHeight(150)
        self.debug_log.setStyleSheet("""
            background: rgba(0, 0, 0, 150);
            color: #00E0E0;
            border: 1px solid #00CCCC;
            border-radius: 5px;
            font-family: 'Consolas', monospace;
            font-size: 10pt;
        """)
        self.debug_log.setVisible(False)
        
        # 设置界面
        self.setup_ui()
        self.setup_connections()
        
        # 预加载YOLO模型
        self.pre_load_yolo_model()
        
        # 启动MQTT
        QTimer.singleShot(1000, self.init_mqtt)
        
        # 定时器设置
        self.setup_timers()
    
    def pre_load_yolo_model(self):
        """预加载YOLO模型，提高响应速度"""
        if YOLO_AVAILABLE:
            try:
                if os.path.exists(YOLO_MODEL_PATH):
                    self.log_message(f"预加载YOLO模型: {YOLO_MODEL_PATH}")
                    self.yolo_detector = YOLODetector(YOLO_MODEL_PATH)
                    model_init_thread = threading.Thread(target=self.yolo_detector.initialize)
                    model_init_thread.daemon = True
                    model_init_thread.start()
                else:
                    self.log_message(f"YOLO模型文件不存在: {YOLO_MODEL_PATH}")
            except Exception as e:
                self.log_message(f"预加载YOLO模型失败: {e}")
    
    def setup_timers(self):
        """设置定时器"""
        # UI更新定时器
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui_time)
        self.ui_timer.start(1000)
        
        # 检测结果更新定时器
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self.update_detection_results)
        self.detection_timer.setInterval(1000)
        self.detection_timer.start()
        
        # 环境分析定时器
        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self.update_situation_analysis)
        self.analysis_timer.setInterval(3000)
        self.analysis_timer.start()
        
        # MQTT连接监控定时器
        self.mqtt_monitor_timer = QTimer()
        self.mqtt_monitor_timer.timeout.connect(self.monitor_mqtt_connection)
        self.mqtt_monitor_timer.setInterval(10000)  # 每10秒检查一次
        self.mqtt_monitor_timer.start()
        
        # GPS状态监控定时器
        self.gps_monitor_timer = QTimer()
        self.gps_monitor_timer.timeout.connect(self.monitor_gps_status)
        self.gps_monitor_timer.setInterval(5000)  # 每5秒检查GPS状态
        self.gps_monitor_timer.start()
    
    def monitor_mqtt_connection(self):
        """监控MQTT连接状态"""
        if hasattr(self, 'mqtt_thread') and self.mqtt_thread:
            if not self.mqtt_thread.is_connected:
                self.mqtt_connection_attempts += 1
                if self.mqtt_connection_attempts > 5:
                    self.log_message("MQTT连接长时间失败，请检查网络和服务器状态")
    
    def monitor_gps_status(self):
        """监控GPS状态"""
        # 检查GPS数据是否在正常更新
        current_time = time.time()
        if current_time - self.last_gps_update > 30:  # 超过30秒没有GPS更新
            if self.gps_update_count > 0:  # 之前有过GPS数据
                self.log_message("GPS信号丢失：超过30秒未收到GPS更新")
    
    def add_control_buttons(self):
        """添加控制按钮到状态栏"""
        # 调试按钮
        self.debug_btn = QPushButton("🔍 调试")
        self.debug_btn.setFixedSize(80, 30)
        self.debug_btn.setStyleSheet("""
            background: rgba(0, 224, 224, 120);
            border: 2px solid #00CCCC;
            border-radius: 5px;
            color: white;
            font-weight: bold;
        """)
        self.debug_btn.clicked.connect(self.toggle_debug_window)
        self.statusBar().addPermanentWidget(self.debug_btn)
    
    def setup_ui(self):
        """设置UI"""
        # 全局样式 - 管道检测风格
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0A1015, stop:0.5 #101820, stop:1 #0A1015);
                color: #FFFFFF;
            }
            QGroupBox {
                font-size: 16pt;
                font-weight: bold;
                color: #00CCCC;
                border: 3px solid #00CCCC;
                border-radius: 5px;
                margin-top: 30px;
                padding-top: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(16, 24, 32, 180),
                    stop:1 rgba(10, 16, 21, 180));
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 25px;
                padding: 0 20px;
                color: #00CCCC;
            }
            /* 定义滚动条样式 */
            QScrollBar:vertical {
                width: 10px;
                background: rgba(16, 24, 32, 150);
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #00CCCC;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 顶部状态栏
        self.create_top_header(main_layout)
        
        # 主内容区 - 两列布局
        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        left_panel.setFixedWidth(400)
        
        # 右侧监控区域
        right_panel = self.create_monitoring_panel()
        
        content_layout.addWidget(left_panel)
        content_layout.addWidget(right_panel, 1)
        
        main_layout.addLayout(content_layout, 1)
        
        # 底部状态栏
        self.create_status_bar()
        
        # 添加调试日志容器
        main_layout.addWidget(self.debug_log)
        self.debug_log.setVisible(False)
    
    def create_top_header(self, layout):
        """创建顶部标题栏"""
        header_frame = QFrame()
        header_frame.setFixedHeight(100)
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 170, 204, 80),
                    stop:0.5 rgba(0, 224, 224, 80),
                    stop:1 rgba(0, 170, 204, 80));
                border: 3px solid #00CCCC;
                border-radius: 5px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(30, 15, 30, 15)
        
        # 主标题
        title_label = QLabel(f"🐍管道检测机器蛇实时控制系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 30pt;
            font-weight: bold;
            color: #FFFFFF;
            letter-spacing: 2px;
        """)
        
        # 布局
        header_layout.addWidget(title_label, 1)
        
        layout.addWidget(header_frame)
    
    def create_control_panel(self):
        """创建控制面板"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(25)
        
        # 模式选择
        mode_group = QGroupBox("🎮 控制模式选择")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(15)
        
        self.mode_buttons = []
        for mode in GAIT_MODES:
            btn = HexagonButton(mode)
            btn.setCheckable(True)
            btn.setFixedHeight(80)
            btn.clicked.connect(lambda checked, m=mode: self.select_mode(m))
            mode_layout.addWidget(btn)
            self.mode_buttons.append(btn)
        
        # 方向控制
        direction_group = QGroupBox("🎯 方向控制")
        direction_widget = QWidget()
        direction_layout = QGridLayout(direction_widget)
        direction_layout.setSpacing(20)
        direction_layout.setContentsMargins(40, 50, 40, 40)
        
        self.direction_buttons = {}
        positions = {
            "前进": (0, 1), "左转": (1, 0), "复位": (1, 1),
            "右转": (1, 2), "后退": (2, 1)
        }
        
        for direction, (row, col) in positions.items():
            btn = TechDirectionButton(direction)
            btn.clicked.connect(lambda checked, d=direction: self.select_direction(d))
            direction_layout.addWidget(btn, row, col, Qt.AlignCenter)
            self.direction_buttons[direction] = btn
        
        direction_group_layout = QVBoxLayout(direction_group)
        direction_group_layout.addWidget(direction_widget)
        
        # 状态显示
        status_group = QGroupBox("📊 系统状态")
        status_layout = QVBoxLayout(status_group)
        
        self.mode_status_label = QLabel("模式: 未选择")
        self.mode_status_label.setStyleSheet("""
            background: rgba(0, 224, 224, 30);
            border: 2px solid rgba(0, 224, 224, 150);
            border-radius: 5px;
            padding: 15px;
            font-size: 32pt;
            font-weight: bold;
            color: #00CCCC;
        """)
        
        self.connection_status_label = QLabel("MQTT: 正在连接...")
        self.connection_status_label.setStyleSheet("""
            background: rgba(255, 204, 0, 30);
            border: 2px solid rgba(255, 204, 0, 150);
            border-radius: 5px;
            padding: 15px;
            font-size: 16pt;
            font-weight: bold;
            color: #FFCC00;
        """)
        
        # 控制按钮
        control_btn_layout = QHBoxLayout()
        
        self.detection_btn = QPushButton("🔍 智能检测")
        self.reset_btn = QPushButton("🔄 重置系统")
        
        for btn in [self.detection_btn, self.reset_btn]:
            btn.setFixedHeight(50)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(0, 224, 224, 80);
                    border: 2px solid #00CCCC;
                    border-radius: 5px;
                    font-size: 14pt;
                    font-weight: bold;
                    color: white;
                    padding: 10px;
                }
                QPushButton:hover {
                    background: rgba(0, 224, 224, 120);
                    border: 2px solid #00E0E0;
                }
            """)
        
        self.detection_btn.clicked.connect(self.toggle_detection)
        self.reset_btn.clicked.connect(self.reset_all)
        
        control_btn_layout.addWidget(self.detection_btn)
        control_btn_layout.addWidget(self.reset_btn)
        
        status_layout.addWidget(self.mode_status_label)
        status_layout.addWidget(self.connection_status_label)
        status_layout.addLayout(control_btn_layout)
        
        control_layout.addWidget(mode_group, 1)
        control_layout.addWidget(direction_group, 1)
        control_layout.addWidget(status_group, 1)
        
        return control_widget
    
    def create_monitoring_panel(self):
        """创建监控面板 - 管道检测风格，实现管道缺陷检测和内部监控并列"""
        monitor_widget = QWidget()
        monitor_layout = QVBoxLayout(monitor_widget)
        monitor_layout.setContentsMargins(0, 0, 0, 20)  # 减少顶部间距
        monitor_layout.setSpacing(20)
        
        # 上方区域 - 管道内部监控和缺陷检测并列
        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        top_layout.setSpacing(15)
        
        # 管道内部监控 - 左侧，较大
        video_group = QGroupBox("📺 管道内部监控")
        
        # 设置管道内部监控组标题的字体大小
        video_group.setStyleSheet("""
            QGroupBox {
                font-size: 30pt;  /* 调整这个值来改变字体大小 */
                font-weight: bold;
                color: #00CCCC;
                border: 3px solid #00CCCC;
                border-radius: 5px;
                margin-top: 30px;
                padding-top: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(16, 24, 32, 180),
                    stop:1 rgba(10, 16, 21, 180));
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 25px;
                padding: 0 20px;
                color: #00CCCC;
            }
        """)
        
        video_layout = QVBoxLayout(video_group)
        
        self.video_widget = PipelineVideoWidget()
        self.video_widget.setMinimumSize(700, 500)
        
        # 视频控制按钮组
        video_controls = QHBoxLayout()
        
        self.toggle_info_btn = QPushButton("📊 信息")
        self.toggle_grid_btn = QPushButton("📏 网格")
        self.toggle_crosshair_btn = QPushButton("➕ 十字线")
        self.screenshot_btn = QPushButton("📷 截图")
        
        video_buttons = [self.toggle_info_btn, self.toggle_grid_btn, 
                        self.toggle_crosshair_btn, self.screenshot_btn]
                        
        for btn in video_buttons:
            btn.setFixedHeight(36)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(0, 204, 204, 80);
                    border: 2px solid #00CCCC;
                    border-radius: 5px;
                    font-size: 12pt;
                    color: white;
                }
                QPushButton:hover {
                    background: rgba(0, 224, 224, 120);
                }
            """)
            
        self.toggle_info_btn.clicked.connect(lambda: self.video_widget.toggle_info())
        self.toggle_grid_btn.clicked.connect(lambda: self.video_widget.toggle_grid())
        self.toggle_crosshair_btn.clicked.connect(lambda: self.video_widget.toggle_crosshair())
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        
        video_controls.addWidget(self.toggle_info_btn)
        video_controls.addWidget(self.toggle_grid_btn)
        video_controls.addWidget(self.toggle_crosshair_btn)
        video_controls.addWidget(self.screenshot_btn)
        
        video_layout.addWidget(self.video_widget)
        video_layout.addLayout(video_controls)
        
        # 管道缺陷检测区域 - 右侧，较小
        defect_group = QGroupBox("🔍 管道缺陷检测")
        defect_group.setStyleSheet("""
            QGroupBox {
                font-size: 30pt;  /* 调整这个值来改变字体大小 */
                font-weight: bold;
                color: #00CCCC;
                border: 3px solid #00CCCC;
                border-radius: 5px;
                margin-top: 30px;
                padding-top: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(16, 24, 32, 180),
                    stop:1 rgba(10, 16, 21, 180));
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 25px;
                padding: 0 20px;
                color: #00CCCC;
            }
        """)
        defect_layout = QVBoxLayout(defect_group)
        
        # 缺陷检测面板
        self.defect_panel = PipelineDefectWidget()
        defect_layout.addWidget(self.defect_panel)
        
        # 设置比例，视频区域更大
        top_layout.addWidget(video_group, 65)  # 65% 的宽度
        top_layout.addWidget(defect_group, 35)  # 35% 的宽度
        
        # 管道检测分析区域 - 底部
        analysis_group = QGroupBox("🧠 管道检测分析")
        analysis_group.setStyleSheet("""
            QGroupBox {
                font-size: 25pt;  /* 调整这个值来改变字体大小 */
                font-weight: bold;
                color: #00CCCC;
                border: 3px solid #00CCCC;
                border-radius: 5px;
                margin-top: 30px;
                padding-top: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(16, 24, 32, 180),
                    stop:1 rgba(10, 16, 21, 180));
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 25px;
                padding: 0 20px;
                color: #00CCCC;
            }
        """)
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                background: rgba(16, 24, 32, 150);
                border: none;
                border-radius: 5px;
                color: #FFFFFF;
                font-size: 12pt;
                padding: 10px;
            }
        """)
        
        analysis_layout.addWidget(self.analysis_text)
        
        # 添加到主监控布局 - 不再包含标题，直接添加监控和分析区域
        monitor_layout.addWidget(top_container, 3)  # 3份空间
        monitor_layout.addWidget(analysis_group, 1)  # 1份空间
        
        return monitor_widget
    
    def take_screenshot(self):
        """截取当前摄像头画面"""
        try:
            frame = self.video_widget.get_current_processed_frame()
            if frame is None:
                self.log_message("无法获取当前帧进行截图")
                return
                
            # 保存截图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_inspection_{timestamp}.jpg"
            
            # 确保screenshots目录存在
            os.makedirs("screenshots", exist_ok=True)
            filepath = os.path.join("screenshots", filename)
            
            # 保存图像
            cv2.imwrite(filepath, frame)
            
            # 在状态栏显示通知
            self.statusBar().showMessage(f"已保存截图: {filepath}", 5000)
            self.log_message(f"成功保存截图: {filepath}")
            
        except Exception as e:
            self.log_message(f"截图保存失败: {e}")
    
    def create_status_bar(self):
        """创建状态栏"""
        status_bar = QStatusBar()
        status_bar.setFixedHeight(50)
        status_bar.setStyleSheet("""
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 170, 204, 80),
                    stop:1 rgba(0, 224, 224, 80));
                color: white;
                border-top: 3px solid #00CCCC;
                font-weight: bold;
                font-size: 12pt;
                padding: 8px;
            }
        """)
        
        # 显示资源管理信息
        resources = self.resource_manager.get_resources_summary()
        gpu_info = "GPU" if resources["gpu_available"] else "CPU"
        status_bar.showMessage(f"🚀 管道检测机器人控制系统就绪 - 使用{gpu_info}处理 - 等待MQTT连接...")
        
        self.setStatusBar(status_bar)
        
        # 添加控制按钮
        self.add_control_buttons()
    
    def setup_connections(self):
        """设置信号连接"""
        # 按钮连接
        self.reset_btn.clicked.connect(self.reset_all)
        
        # 视频流信号
        if hasattr(self, 'video_widget'):
            self.video_widget.frame_ready.connect(self.process_video_frame)
    
    def update_ui_time(self):
        """更新时间显示"""
        if hasattr(self, 'time_label'):
            self.time_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def update_connection_status(self, connected):
        """更新连接状态"""
        self.mqtt_connected = connected
        
        if connected:
            self.mqtt_connection_attempts = 0  # 重置连接尝试计数
            
            self.connection_status_label.setText("MQTT: 已连接 ✅")
            self.connection_status_label.setStyleSheet("""
                background: rgba(0, 224, 224, 30);
                border: 2px solid rgba(0, 224, 224, 150);
                border-radius: 5px;
                padding: 15px;
                font-size: 16pt;
                font-weight: bold;
                color: #00E0E0;
            """)
            
            resources = self.resource_manager.get_resources_summary()
            processor_type = "GPU" if resources["gpu_available"] else "CPU"
            self.statusBar().showMessage(f"🌐 MQTT连接成功 - 管道检测机器人控制系统就绪 - 使用{processor_type}处理")
            
        else:
            self.connection_status_label.setText("MQTT: 连接失败 ❌")
            self.connection_status_label.setStyleSheet("""
                background: rgba(255, 85, 85, 30);
                border: 2px solid rgba(255, 85, 85, 150);
                border-radius: 5px;
                padding: 15px;
                font-size: 16pt;
                font-weight: bold;
                color: #FF5555;
            """)
            
            self.statusBar().showMessage("❌ MQTT连接失败 - 请检查网络设置和服务器状态")
    
    def handle_gps_data(self, gps_data):
        """处理GPS数据 - 管道检测版本"""
        try:
            # 详细日志输出接收到的GPS数据
            self.log_message(f"收到GPS数据: lat={gps_data.get('latitude', 'N/A')}, lng={gps_data.get('longitude', 'N/A')}")
            
            # 确保有有效的GPS数据
            valid_flag = gps_data.get("valid", gps_data.get("is_valid", False))
            
            if not valid_flag:
                self.log_message("GPS数据标记为无效")
                return
            
            # 提取并验证GPS数据
            try:
                lat = float(gps_data.get("latitude", 0))
                lng = float(gps_data.get("longitude", 0))
                
                # 验证坐标有效性
                if abs(lat) > 90 or abs(lng) > 180:
                    self.log_message(f"GPS坐标超出范围: lat={lat}, lng={lng}")
                    return
                
                # 排除明显无效坐标
                if lat == 0 and lng == 0:
                    self.log_message("GPS坐标为(0,0)，跳过更新")
                    return
                
                # 提取其他GPS信息
                accuracy = float(gps_data.get("accuracy", 999))
                satellites = int(gps_data.get("satellites", 0))
                signal_strength = float(gps_data.get("signal_strength", 0))
                altitude = float(gps_data.get("altitude", 0))
                speed = float(gps_data.get("speed", 0))
                
                self.log_message(f"GPS数据解析成功: lat={lat:.6f}, lng={lng:.6f}, satellites={satellites}")
                
            except (ValueError, TypeError) as e:
                self.log_message(f"GPS数据格式错误: {e}")
                return
                
            # 更新数据融合系统
            self.data_fusion.update_gps_data(gps_data)
            
            # 记录最后更新时间和计数
            self.last_gps_update = time.time()
            self.gps_update_count += 1
                
        except Exception as e:
            self.log_message(f"处理GPS数据时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    def select_mode(self, mode):
        """选择运动模式"""
        # 取消所有按钮选中状态
        for btn in self.mode_buttons:
            btn.setChecked(False)
        
        if self.current_mode == mode:
            # 如果点击的是当前模式，则取消选择
            self.current_mode = None
            self.mode_status_label.setText("模式: 未选择")
            self.statusBar().showMessage("❌ 已取消模式选择")
        else:
            # 选择新模式
            for btn in self.mode_buttons:
                if btn.text() == mode:
                    btn.setChecked(True)
                    break
            
            self.current_mode = mode
            self.mode_status_label.setText(f"模式: {mode}")
            self.statusBar().showMessage(f"✅ 已选择运动模式: {mode}")
            
            # 如果是复位模式，直接发送命令
            if mode == "复位模式":
                self.send_robot_command({"mode": mode, "direction": "复位"})
    
    def select_direction(self, direction):
        """选择移动方向"""
        # 更新方向按钮状态
        for name, btn in self.direction_buttons.items():
            btn.setChecked(name == direction)
        
        self.current_direction = direction
        
        # 检查是否已选择模式
        if not self.current_mode:
            QMessageBox.warning(self, "警告", 
                "⚠️ 请先选择运动模式！\n\n需要选择蠕动模式或蜿蜒模式后才能控制方向。")
            # 清除方向选择
            for btn in self.direction_buttons.values():
                btn.setChecked(False)
            return
        
        if self.current_mode == "复位模式":
            QMessageBox.information(self, "提示", 
                "ℹ️ 复位模式不支持方向控制\n\n复位模式会自动执行复位动作。")
            return
        
        # 发送控制命令
        command = {
            "mode": self.current_mode,
            "direction": direction,
            "timestamp": time.time(),
            "user": CURRENT_USER
        }
        
        success = self.send_robot_command(command)
        if success:
            self.statusBar().showMessage(f"🎯 发送控制命令: {self.current_mode} - {direction}")
        else:
            self.statusBar().showMessage("❌ 命令发送失败 - 请检查MQTT连接")
    
    def send_robot_command(self, command):
        """发送机器人控制命令"""
        if hasattr(self, 'mqtt_thread') and self.mqtt_thread.is_connected:
            return self.mqtt_thread.publish_command(command)
        return False
    
    def toggle_detection(self):
        """切换智能检测功能 - 使用yolo11模型"""
        if self.video_widget.detection_enabled:
            # 如果已启用，则禁用
            self.video_widget.enable_yolo_detection(False)
            self.detection_btn.setText("🔍 智能检测")
            self.statusBar().showMessage("已关闭智能缺陷检测")
            self.defect_panel.update_model_info(YOLO_MODEL_PATH, ["已关闭"])
        else:
            # 如果禁用，尝试启用 - 使用本地模型
            success = self.video_widget.enable_yolo_detection(True)
            if success:
                self.detection_btn.setText("🔍 关闭检测")
                self.statusBar().showMessage("已开启智能缺陷检测")
                
                # 更新缺陷面板模型信息
                if self.video_widget.yolo_detector and hasattr(self.video_widget.yolo_detector, 'names'):
                    self.defect_panel.update_model_info(
                        YOLO_MODEL_PATH, 
                        list(self.video_widget.yolo_detector.names.values())
                    )
            else:
                QMessageBox.warning(
                    self, 
                    "警告", 
                    f"智能检测初始化失败\n\n请确保YOLO模型文件'{YOLO_MODEL_PATH}'存在且有效，并已安装YOLO库"
                )
    
    def handle_sensor_data(self, data):
        """处理传感器数据 - 管道检测版本"""
        try:
            # 更新数据融合系统
            self.data_fusion.update_sensor_data(data)
            
            # 检查管道缺陷信息
            if "pipeline_defects" in data:
                self.defect_panel.update_defects(data["pipeline_defects"])
                self.data_fusion.update_pipeline_defects(data["pipeline_defects"])
                
            # 检查管道信息
            if "pipeline_info" in data:
                self.defect_panel.update_pipeline_info(data["pipeline_info"])
            
            # 处理传感器数据中的GPS信息
            if self._process_gps_from_sensor_data(data):
                # GPS数据已处理
                pass
            
            # 在视频中显示检测到的缺陷
            if "pipeline_defects" in data and hasattr(self, 'video_widget'):
                self.video_widget.update_defect_data(data["pipeline_defects"])
            
        except Exception as e:
            self.log_message(f"处理传感器数据错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_gps_from_sensor_data(self, data):
        """从传感器数据中处理GPS信息"""
        if ("gps_valid" in data and data["gps_valid"] and 
            "latitude" in data and "longitude" in data):
            
            try:
                lat = float(data["latitude"])
                lng = float(data["longitude"])
                
                if abs(lat) <= 90 and abs(lng) <= 180 and (lat != 0 or lng != 0):
                    # 构建GPS数据包
                    gps_data = {
                        "latitude": lat,
                        "longitude": lng,
                        "altitude": float(data.get("altitude", 0)),
                        "speed": float(data.get("gps_speed", data.get("speed", 0))),
                        "course": float(data.get("gps_course", data.get("course", 0))),
                        "satellites": int(data.get("gps_satellites", data.get("satellites", 0))),
                        "signal_strength": float(data.get("gps_signal_strength", data.get("signal_strength", 0))),
                        "accuracy": float(data.get("gps_accuracy", data.get("accuracy", 999))),
                        "timestamp": time.time(),
                        "valid": True
                    }
                    
                    # 直接处理GPS数据
                    self.handle_gps_data(gps_data)
                    return True
                    
            except (ValueError, TypeError) as e:
                self.log_message(f"传感器数据中的GPS格式错误: {e}")
                
        return False
    
    def process_camera_frame(self, cv_frame):
        """处理从MQTT接收的摄像头帧"""
        try:
            if cv_frame is None or cv_frame.size == 0:
                return
                
            # 更新视频显示
            self.video_widget.update_frame(cv_frame)
            
            # 如果启用了YOLO检测，对帧进行处理
            if self.video_widget.detection_enabled and self.video_widget.yolo_detector:
                # 通过视频widget的更新功能进行处理，无需额外处理
                pass
                
        except Exception as e:
            self.log_message(f"处理摄像头帧错误: {e}")
    
    def process_video_frame(self, cv_frame):
        """处理视频帧并更新数据融合系统的检测结果"""
        # 只在检测启用且有帧时进行处理
        if self.video_widget.detection_enabled and cv_frame is not None:
            try:
                # 获取YOLO检测结果（不再重复检测，从视频控件获取）
                if self.video_widget.yolo_detector and hasattr(self.video_widget, 'current_processed_frame'):
                    # 如果已有处理后的帧，从视频控件获取检测结果
                    detections = []
                    if hasattr(self.video_widget.yolo_detector, 'last_detections'):
                        detections = self.video_widget.yolo_detector.last_detections
                    
                    # 更新数据融合系统
                    self.data_fusion.update_yolo_detections(detections)
                    
                    # 更新管道缺陷显示
                    self.defect_panel.update_defects(self.data_fusion.get_pipeline_defects())
                    
            except Exception as e:
                self.log_message(f"处理检测结果错误: {e}")
    
    def update_detection_results(self):
        """更新检测结果显示"""
        try:
            # 获取管道缺陷数据
            pipeline_defects = self.data_fusion.get_pipeline_defects()
            pipeline_info = self.data_fusion.get_pipeline_info()
            
            # 更新检测结果面板
            self.defect_panel.update_defects(pipeline_defects)
            self.defect_panel.update_pipeline_info(pipeline_info)
            
        except Exception as e:
            self.log_message(f"更新检测结果错误: {e}")
    
    def update_situation_analysis(self):
        """更新环境分析结果 - 保留原始实现"""
        try:
            # 获取情境分析结果
            analysis = self.data_fusion.analyze_situation()
            
            if not analysis:
                return
                
            # 生成分析报告
            report = f"""
            <!-- 分别设置每个元素的字体大小 -->
            <h2 style="font-size: 24pt; color: #00CCCC; text-align: center; margin: 25px 0;">管道检测实时分析报告</h2>
            
            <div style="background-color: rgba(0, 204, 204, 0.1); padding: 15px; border-left: 5px solid #00CCCC; margin: 15px 0;">
                <h3 style="font-size: 24pt; margin-top: 10px; margin-bottom: 10px;">管道状态评估</h3>
                <p style="font-size: 24pt; font-weight: bold; color: {
                    '#00E0E0' if analysis['risk_level'] == '低' else
                    '#FFCC00' if analysis['risk_level'] == '中' else '#FF5555'
                };">{analysis['summary']}</p>
                <p style="font-size: 24pt; margin: 10px 0;">风险等级: <span style="font-weight: bold; font-size: 18pt; color: {
                    '#00E0E0' if analysis['risk_level'] == '低' else
                    '#FFCC00' if analysis['risk_level'] == '中' else '#FF5555'
                };">{analysis['risk_level']}</span></p>
            </div>
            """
            
            # 添加管道信息
            pipe_info = self.data_fusion.get_pipeline_info()
            if pipe_info:
                report += f"""
                <div style="background-color: rgba(0, 170, 204, 0.1); padding: 15px; border-left: 5px solid #00AACC; margin: 15px 0;">
                    <h3 style="font-size: 20pt; margin-top: 10px; margin-bottom: 10px;">管道基本信息</h3>
                    <ul style="font-size: 24pt; margin: 10px 0; padding-left: 25px;">
                        <li style="font-size: 22pt; margin: 5px 0;">管道直径: {pipe_info.get('diameter', 0)} mm</li>
                        <li style="font-size: 22pt; margin: 5px 0;">管道材质: {pipe_info.get('material', '未知')}</li>
                        <li style="font-size: 22pt; margin: 5px 0;">管道介质: {pipe_info.get('medium', '未知')}</li>
                        <li style="font-size: 22pt; margin: 5px 0;">检测距离: {pipe_info.get('distance', 0):.1f} m</li>
                        <li style="font-size: 22pt; margin: 5px 0;">缺陷总数: {pipe_info.get('defect_count', 0)} 处</li>
                    </ul>
                </div>
                """
                
            # 添加YOLO检测信息
            detections = self.data_fusion.get_yolo_detections()
            if detections:
                # 按照类别统计检测数量
                detection_counts = {}
                for det in detections:
                    cls_name = det.get('name', '未知')
                    if cls_name in detection_counts:
                        detection_counts[cls_name] += 1
                    else:
                        detection_counts[cls_name] = 1
                
                # 生成检测统计报告
                report += f"""
                <div style="background-color: rgba(0, 224, 224, 0.1); padding: 15px; border-left: 5px solid #00E0E0; margin: 15px 0;">
                    <h3 style="font-size: 20pt; margin-top: 10px; margin-bottom: 10px;">智能检测统计</h3>
                    <ul style="font-size: 16pt; margin: 10px 0; padding-left: 25px;">
                """
                for cls_name, count in detection_counts.items():
                    report += f'<li style="font-size: 16pt; margin: 5px 0;">{cls_name}: {count}个</li>'
                report += "</ul></div>"
            
            # 添加关键发现
            if analysis["critical_findings"]:
                report += f"""
                <div style="background-color: rgba(255, 85, 85, 0.1); padding: 15px; border-left: 5px solid #FF5555; margin: 15px 0;">
                    <h3 style="font-size: 20pt; margin-top: 10px; margin-bottom: 10px;">关键发现</h3>
                    <ul style="font-size: 16pt; margin: 10px 0; padding-left: 25px;">
                """
                for finding in analysis["critical_findings"]:
                    report += f'<li style="font-size: 16pt; margin: 5px 0;">{finding}</li>'
                report += "</ul></div>"
            
            # 添加建议操作
            if analysis["recommendations"]:
                report += f"""
                <div style="background-color: rgba(0, 224, 224, 0.1); padding: 15px; border-left: 5px solid #00E0E0; margin: 15px 0;">
                    <h3 style="font-size: 20pt; margin-top: 10px; margin-bottom: 10px;">建议操作</h3>
                    <ul style="font-size: 16pt; margin: 10px 0; padding-left: 25px;">
                """
                for rec in analysis["recommendations"]:
                    report += f'<li style="font-size: 16pt; margin: 5px 0;">{rec}</li>'
                report += "</ul></div>"
            
            # 添加GPS定位信息
            gps_data = self.data_fusion.get_gps_data()
            if gps_data and self.data_fusion.get_gps_valid():
                report += f"""
                <div style="background-color: rgba(0, 170, 204, 0.1); padding: 15px; border-left: 5px solid #00AACC; margin: 15px 0;">
                    <h3 style="font-size: 20pt; margin-top: 10px; margin-bottom: 10px;">实时定位信息</h3>
                    <p style="font-size: 16pt; margin: 10px 0;">经度: {gps_data.get('longitude', 0):.6f}° &nbsp; 纬度: {gps_data.get('latitude', 0):.6f}°</p>
                    <p style="font-size: 16pt; margin: 10px 0;">卫星数量: {gps_data.get('satellites', 0)}颗 &nbsp; 
                    信号强度: {gps_data.get('signal_strength', 0):.0f}% &nbsp;
                    定位精度: {gps_data.get('accuracy', 999):.1f}m</p>
                </div>
                """
                    
            # 更新分析文本
            self.analysis_text.setHtml(report)
            
        except Exception as e:
            self.log_message(f"更新环境分析错误: {e}")
    
    def reset_all(self):
        """重置所有控制状态"""
        # 取消所有按钮选中状态
        for btn in self.mode_buttons:
            btn.setChecked(False)
        for btn in self.direction_buttons.values():
            btn.setChecked(False)
        
        # 重置状态
        self.current_mode = None
        self.current_direction = None
        
        # 更新显示
        self.mode_status_label.setText("模式: 未选择")
        
        # 发送复位命令
        success = self.send_robot_command({"mode": "复位模式", "direction": "复位"})
        
        if success:
            self.statusBar().showMessage("✅ 已发送复位命令，所有状态已重置")
        else:
            self.statusBar().showMessage("❌ 复位命令发送失败 - 请检查MQTT连接")
    
    def log_message(self, message):
        """记录消息到调试日志"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        
        if hasattr(self, 'debug_log'):
            self.debug_log.append(log_entry)
            # 自动滚动到底部
            cursor = self.debug_log.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.debug_log.setTextCursor(cursor)
    
    def toggle_debug_window(self):
        """切换调试窗口显示状态"""
        self.debug_log.setVisible(not self.debug_log.isVisible())
        
        if self.debug_log.isVisible():
            self.debug_btn.setText("🔍 关闭调试")
        else:
            self.debug_btn.setText("🔍 调试")
    
    def init_mqtt(self):
        """初始化MQTT连接"""
        try:
            self.log_message("开始初始化MQTT连接...")
            
            # 创建MQTT线程
            self.mqtt_thread = MQTTThread()
            
            # 连接信号
            self.mqtt_thread.sensor_data_signal.connect(self.handle_sensor_data)
            self.mqtt_thread.gps_data_signal.connect(self.handle_gps_data)
            self.mqtt_thread.camera_normal_frame_signal.connect(self.process_camera_frame)
            self.mqtt_thread.connection_signal.connect(self.update_connection_status)
            self.mqtt_thread.log_signal.connect(self.log_message)
            
            # 启动线程
            self.mqtt_thread.start()
            
            # 检查连接状态
            QTimer.singleShot(5000, self.check_mqtt_connection)
            
            self.log_message("MQTT线程已启动")
            
        except Exception as e:
            self.log_message(f"MQTT初始化失败: {e}")
    
    def check_mqtt_connection(self):
        """检查MQTT连接状态"""
        if not hasattr(self, 'mqtt_thread') or not self.mqtt_thread.is_connected:
            self.log_message("MQTT连接失败或超时，请检查网络连接")
            self.update_connection_status(False)
    
    def closeEvent(self, event):
        """关闭窗口时的处理"""
        # 停止MQTT线程
        if hasattr(self, 'mqtt_thread'):
            self.mqtt_thread.stop_thread()
            self.mqtt_thread.wait()
        
        # 释放摄像头资源
        if hasattr(self, 'video_widget') and hasattr(self.video_widget, 'release'):
            self.video_widget.release()
            
        # 确保发送复位命令
        try:
            if hasattr(self, 'mqtt_thread') and self.mqtt_thread.is_connected:
                self.mqtt_thread.publish_command({"mode": "复位模式", "direction": "复位"})
                QThread.msleep(500)  # 等待命令发送完成
        except:
            pass
            
        # 接受关闭事件
        event.accept()

def main():
    """主函数"""
    # 更新日期和用户信息
    global CURRENT_DATE, CURRENT_USER
    CURRENT_DATE = "2025-08-25 07:52:21"
    CURRENT_USER = "12ljf"
    
    print(f"启动管道检测机器人控制系统 - 用户: {CURRENT_USER}, 时间: {CURRENT_DATE}")
    
    # 启动应用
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 加载自定义字体
    fonts = []
    for font_family in ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Microsoft YaHei"]:
        font = QFont(font_family)
        if QFontInfo(font).exactMatch():
            app.setFont(font)
            print(f"使用字体: {font_family}")
            break
    
    # 创建主窗口
    window = PipelineRobotDashboard()
    window.show()
    
    # 启动事件循环
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
