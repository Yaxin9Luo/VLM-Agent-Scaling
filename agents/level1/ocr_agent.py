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
                "Your ONLY task is to read and extract text from images (or video frames) with extremely high attention to detail.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. FOCUS: Examine every pixel for any possible text.\n"
                "2. THOROUGHNESS: Scan multiple times to ensure no text is missed.\n"
                "3. SENSITIVITY: Extract faint, small, or partially visible text.\n"
                "4. ACCURACY: Transcribe text exactly, including:\n"
                "   - Numbers, dates, and punctuation\n"
                "   - Special characters and unusual fonts\n"
                "   - Case sensitivity and spacing\n"
                "5. FORMAT: Maintain original layout and line breaks if possible.\n"
                "6. COMPLETENESS: Report all text found, even a single character.\n"
                "7. VERIFICATION: Double-check for errors or missed text.\n"
                "8. NO_TEXT_FOUND: Use this response only if you are 100% certain no text is present.\n\n"
                "CHAIN OF THOUGHT / REFLECTION / RETHINKING:\n"
                " - Think step by step: carefully gather text candidates from the image.\n"
                " - Reflect on faint or ambiguous text. Re-scan if uncertain.\n"
                " - Provide final text output only (do not describe the image).\n\n"
                "IMPORTANT: Your primary goal is text extraction. Do not add extra info.\n"
                "Do NOT repeat the user question or instructions in your final output.\n"
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