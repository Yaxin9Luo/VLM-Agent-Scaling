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
        logger.debug(f"Original image size: {image.size}, mode: {image.mode}")
        
        # 只保留RGB转换，移除其他所有预处理
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.debug("Converted image to RGB mode")
        
        return image

    def process(self, input_data: AgentInput) -> AgentOutput:
        """处理图像OCR请求"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires image_path")

        try:
            # 读取图片
            logger.info(f"Processing image: {input_data.image_path}")
            image = Image.open(input_data.image_path)
            logger.debug(f"Successfully loaded image: size={image.size}, mode={image.mode}")
            
            image = self.enhance_for_ocr(image)
            
            # 准备提示词
            system_prompt = (
                "You are a specialized OCR (Optical Character Recognition) system. "
                "Your ONLY task is to read and extract text from images with extremely high attention to detail. "
                "Critical Instructions:\n"
                "1. FOCUS: Look at every pixel of the image for ANY text\n"
                "2. THOROUGHNESS: Scan the entire image multiple times to ensure no text is missed\n"
                "3. SENSITIVITY: Extract even faint, small, or partially visible text\n"
                "4. ACCURACY: Report text exactly as it appears, including:\n"
                "   - Numbers and dates\n"
                "   - Punctuation marks\n"
                "   - Special characters\n"
                "   - Different fonts and styles\n"
                "5. FORMAT: Maintain the original text layout and structure\n"
                "6. COMPLETENESS: If you see ANY text at all, even a single character, you MUST report it\n"
                "7. VERIFICATION: Double-check your findings before responding\n"
                "8. NO_TEXT_FOUND: Only use this response if you are 100% certain there is no text after multiple thorough scans\n"
                "\nIMPORTANT: Your primary goal is text extraction. Do not describe the image or provide any other information.\n"
                "DO NOT repeat the question or instructions in your response."
            )

            # 使用InternVL2-8B模型进行推理
            logger.debug("Running inference with InternVL2-8B model")
            query = (
                "Extract and list ALL text content visible in this image. "
                "Include every character, number, and symbol you can find. "
                "If there is no text at all, respond with 'NO_TEXT_FOUND'."
            )
            extracted_text = self.vlm.process_image_query(image, query, system_prompt)
            logger.info(f"Extracted text result: {extracted_text[:200]}...")  # 显示更多文本用于调试

            # 计算置信度分数
            confidence = 0.9 if extracted_text and extracted_text.strip() != "NO_TEXT_FOUND" else 0.1
            logger.info(f"OCR confidence score: {confidence}")

            return AgentOutput(
                result=extracted_text,
                confidence=confidence,
                metadata={
                    "model": "internvl2-8b",
                    "image_size": image.size,
                    "image_mode": image.mode,
                }
            )
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise Exception(f"OCR processing failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.image_path) 