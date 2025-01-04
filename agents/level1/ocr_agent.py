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

    def clean_ocr_text(self, text: str) -> str:
        """
        进行简单的OCR文本清理:
        1. 去除首尾空格
        2. 可根据需要去除重复行、特殊字符等
        """
        # 这是最简洁的示例，你可以按需扩展
        cleaned_text = text.strip()
        return cleaned_text

    def process(self, input_data: AgentInput) -> AgentOutput:
        """处理图像OCR请求"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires image_path")

        try:
            # 读取图片
            logger.info(f"Processing image: {input_data.image_path}")
            image = Image.open(input_data.image_path)
            logger.debug(f"Successfully loaded image: size={image.size}, mode={image.mode}")
            
            # 进行图像增强（仅转换为RGB）
            image = self.enhance_for_ocr(image)
            
            # 准备提示词
            system_prompt = (
                "You are a specialized OCR (Optical Character Recognition) system. "
                "Your main task is to read and extract text from images (or video frames) with extremely high attention to detail.\n\n"

                "However, you are also capable of giving a brief contextual note about the extracted text—"
                "for instance, if the text appears to be part of a math equation, signage, label, or anything that might hint at its usage in the image.\n\n"

                "CRITICAL INSTRUCTIONS:\n"
                "1. FOCUS: Examine every pixel for any possible text.\n"
                "2. THOROUGHNESS: Scan multiple times to ensure no text is missed.\n"
                "3. SENSITIVITY: Extract faint, small, or partially visible text.\n"
                "4. ACCURACY: Transcribe text exactly, including:\n"
                "   - Numbers, dates, and punctuation\n"
                "   - Special characters and unusual fonts\n"
                "   - Case sensitivity and spacing\n"
                "5. FORMAT & LAYOUT: Preserve line breaks and spacing to reflect the original layout as closely as possible.\n"
                "6. CONTEXT: Provide a short note (1-2 sentences) describing how the text seems to be arranged or used in the image "
                "(e.g., 'part of a math formula', 'signage on the top-left corner', 'table column headers', etc.). "
                "If you are unsure, you may say so.\n"
                "7. COMPLETENESS: Report all text found, even a single character.\n"
                "8. VERIFICATION: Double-check for errors or missed text.\n"
                "9. NO_TEXT_FOUND: Use this response only if you are 100% certain no text is present.\n\n"

                "CHAIN OF THOUGHT / REFLECTION / RETHINKING:\n"
                " - Think step by step: carefully gather text candidates from the image.\n"
                " - Reflect on faint or ambiguous text. Re-scan if uncertain.\n"
                " - Provide final text output plus a short context note. Do not fully describe the entire image.\n\n"

                "IMPORTANT: Your primary goal is accurate text extraction and minimal contextual note. "
                "Do not add extra info unrelated to the text. "
                "Do NOT repeat the user question or these instructions in your final output.\n\n"

                "## Reference Examples:\n"
                "1) **Multiple Math Equations**\n"
                "   - Image text (spread over rows/columns):\n"
                "       4 + 7 =    2 + 2 =\n"
                "       7 + 2 =    6 + 1 =\n"
                "       9 + 3 =    3 + 8 =\n"
                "   - Final Output:\n"
                "       4 + 7 =    2 + 2 =\n"
                "       7 + 2 =    6 + 1 =\n"
                "       9 + 3 =    3 + 8 =\n"
                "     Context Note: These lines appear to be math equations arranged in a two-column layout.\n\n"

                "2) **Signage / Poster**\n"
                "   - If the sign reads: \"WELCOME TO PARK\", with some stylized font.\n"
                "   - Final Output:\n"
                "       WELCOME TO PARK\n"
                "     Context Note: This text appears on a large sign or poster, possibly an entrance sign.\n\n"

                "3) **Table with Headers**\n"
                "   - Suppose the image has a small table:\n"
                "       Name   |  Score\n"
                "       Alice  |   90\n"
                "       Bob    |   85\n"
                "   - Final Output:\n"
                "       Name   |  Score\n"
                "       Alice  |   90\n"
                "       Bob    |   85\n"
                "     Context Note: These lines look like a table with columns for Name and Score.\n\n"

                "4) **Label / Tag**\n"
                "   - If the image is a clothing label that reads: \"Machine Wash Cold, 100% Cotton\".\n"
                "   - Final Output:\n"
                "       Machine Wash Cold, 100% Cotton\n"
                "     Context Note: Possibly a garment care label inside clothing.\n\n"

                "5) **No Text Found**\n"
                "   - If there truly is no text at all.\n"
                "   - Final Output:\n"
                "       NO_TEXT_FOUND\n"
                "     Context Note: (No note needed if there's absolutely no text, or you can say 'No text or symbols detected.')\n\n"
            )


            query = (
                "Extract all text content visible in this image **line by line**, preserving spacing "
                "and layout to the extent possible. If columns are present, approximate them with spacing. "
                "If there is no text at all, respond with 'NO_TEXT_FOUND'."
            )

            # 使用InternVL2-8B模型进行推理
            logger.debug("Running OCR inference with InternVL2-8B model")
            raw_extracted_text = self.vlm.process_image_query(image, query, system_prompt)
            logger.info(f"Raw extracted text (first 200 chars): {raw_extracted_text[:200]}...")

            # 进行简单的OCR文本清理
            extracted_text = self.clean_ocr_text(raw_extracted_text)
            logger.info(f"Cleaned OCR text (first 200 chars): {extracted_text[:200]}...")

            # 计算置信度分数
            confidence = 0.9 if extracted_text and extracted_text != "NO_TEXT_FOUND" else 0.1
            logger.info(f"OCR confidence score: {confidence}")

            # 返回AgentOutput，并带上agent_name用于后续Agent识别
            return AgentOutput(
                result=extracted_text,
                confidence=confidence,
                metadata={
                    "agent_name": "ocr",  # <-- 用于后续 CommonAgent 识别该输出
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