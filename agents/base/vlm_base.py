from typing import Optional, Dict, Any
from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from loguru import logger
import torch

class VLMBase:
    """基础VLM类，管理InternVL2-8B模型实例"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLMBase, cls).__new__(cls)
            try:
                # 使用本地模型路径初始化InternVL2-8B模型
                model_path = "/root/autodl-tmp/InternVL2-8B"
                logger.info(f"Initializing model from path: {model_path}")
                
                # 检查CUDA可用性
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    gpu_id = 0  # 使用第一个GPU
                    torch.cuda.set_device(gpu_id)
                    logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                else:
                    logger.warning("No GPU available, using CPU")

                # 配置模型参数并初始化模型
                logger.info("Creating model pipeline...")
                cls._model = pipeline(
                    model_path,
                    backend_config=TurbomindEngineConfig(session_len=8192)
                )
                
                # 验证模型是否正确加载
                if cls._model is None:
                    raise RuntimeError("Model pipeline returned None")
                
                logger.info(f"InternVL2-8B model initialized successfully from {model_path}")
                
                # 尝试进行一个简单的测试查询
                logger.info("Performing test query...")
                test_image = load_image(model_path + "/examples/demo.jpg")
                test_response = cls._model(("What color is this image?", test_image))
                logger.info(f"Test query successful, response: {test_response}")
                
            except Exception as e:
                logger.error(f"Failed to initialize InternVL2-8B model: {str(e)}")
                logger.exception("Full initialization error traceback:")
                raise
        return cls._instance

    @classmethod
    def get_model(cls):
        """获取模型实例"""
        if cls._instance is None:
            cls()
        return cls._model

    @staticmethod
    def process_image_query(image: Image.Image, query: str, system_prompt: Optional[str] = None) -> str:
        """处理图像查询"""
        try:
            model = VLMBase.get_model()
            if model is None:
                raise RuntimeError("Failed to get model instance")
            
            # 使用load_image处理图像
            if isinstance(image, str):
                # 如果输入是图片路径
                processed_image = load_image(image)
            else:
                # 如果输入是PIL.Image对象，先保存为临时文件再加载
                temp_path = "/tmp/temp_image.png"
                image.save(temp_path)
                processed_image = load_image(temp_path)
            
            # 构建查询
            if system_prompt:
                # 如果有系统提示词，将其添加到查询中
                full_query = f"{system_prompt}\n{query}"
            else:
                full_query = query
            
            # 使用简化的调用方式
            logger.debug(f"Calling model with query: {full_query}")
            response = model((full_query, processed_image))
            logger.debug(f"Got response: {response}")
            
            # 从Response对象中获取text属性
            if hasattr(response, 'text'):
                return response.text.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                # 尝试转换为字符串
                return str(response).strip()
            
        except Exception as e:
            logger.error(f"Error processing query with VLM: {str(e)}")
            logger.exception("Full traceback:")
            raise 