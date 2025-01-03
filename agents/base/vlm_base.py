from typing import Optional, Dict, Any
from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig
from loguru import logger

class VLMBase:
    """基础VLM类，管理InternVL2-8B模型实例"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLMBase, cls).__new__(cls)
            try:
                # 初始化InternVL2-8B模型
                cls._model = pipeline(
                    "OpenGVLab/InternVL2-8B",
                    backend_config=TurbomindEngineConfig(session_len=8192)
                )
                logger.info("InternVL2-8B model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize InternVL2-8B model: {str(e)}")
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
        model = VLMBase.get_model()
        try:
            if system_prompt:
                response = model.chat((query, image), system=system_prompt)
            else:
                response = model.chat((query, image))
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error processing query with VLM: {str(e)}")
            raise 