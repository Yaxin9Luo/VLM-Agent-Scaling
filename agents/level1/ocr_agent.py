import os
from typing import Dict, Any, Optional
from PIL import Image, ImageOps, ImageEnhance
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput
from ..base.vlm_base import VLMBase

class OCRAgent(BaseAgent):
    """OCR Agent实现，使用InternVL2-8B模型进行OCR识别"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vlm = VLMBase()
        logger.info("OCR Agent initialized successfully")

    def enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """增强图像以提高OCR质量"""
        # 转换为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 自动对比度
        image = ImageOps.autocontrast(image, cutoff=1)
        
        # 锐化
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # 增加对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        return image

    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理图像OCR请求"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires image_path")

        try:
            # 读取图片
            image = Image.open(input_data.image_path)
            image = self.enhance_for_ocr(image)
            
            # 准备提示词
            system_prompt = (
                "Extract ALL text from the image, preserving the original formatting when possible. "
                "Pay special attention to numbers, dates, and tabular data. "
                "If no text can be found, return 'NO_TEXT_FOUND'. "
            )
            
            if input_data.question:
                system_prompt += f"Focus on text that is relevant to answering the following question: {input_data.question}"

            # 使用InternVL2-8B模型进行推理
            logger.debug("Running inference with InternVL2-8B model")
            query = "Extract all text from this image, maintaining the original structure and formatting:"
            extracted_text = self.vlm.process_image_query(image, query, system_prompt)
            logger.info("Successfully extracted text from image")

            # 计算置信度分数
            confidence = 0.9 if extracted_text and extracted_text != "NO_TEXT_FOUND" else 0.1

            return AgentOutput(
                result=extracted_text,
                confidence=confidence,
                metadata={
                    "model": "internvl2-8b",
                }
            )
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise Exception(f"OCR processing failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.image_path) 