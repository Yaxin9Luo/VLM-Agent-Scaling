import os
import base64
import io
from typing import Dict, Any, Optional
from openai import OpenAI
from PIL import Image, ImageOps, ImageEnhance
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput

class OCRAgent(BaseAgent):
    """OCR Agent实现，使用OpenAI GPT-4 Vision API进行OCR识别"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key)
        logger.info("OCR Agent initialized successfully")

    def enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """增强图像以提高OCR质量"""
        # 转换为RGB模式（OpenAI API需要彩色图像）
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

    def compress_to_limit(self, image: Image.Image, max_bytes: int = 20000000) -> str:
        """压缩图像到指定大小限制（OpenAI允许更大的图片）"""
        # 尝试不同的尺寸和质量组合
        sizes = [2048, 1600, 1200, 800]
        qualities = [95, 85, 75, 65]
        
        for size in sizes:
            # 调整大小
            ratio = size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            resized = image.resize(new_size, Image.Resampling.LANCZOS)
            
            for quality in qualities:
                output = io.BytesIO()
                resized.save(output, format='JPEG', quality=quality, optimize=True)
                data = output.getvalue()
                
                if len(data) <= max_bytes:
                    logger.debug(f"Found acceptable size with dimensions {new_size} and quality {quality}")
                    return base64.b64encode(data).decode()
        
        # 如果还是太大，使用最小的尺寸和质量
        output = io.BytesIO()
        final_size = (800, int(800 * image.size[1] / image.size[0]))
        final_image = image.resize(final_size, Image.Resampling.LANCZOS)
        final_image.save(output, format='JPEG', quality=65, optimize=True)
        return base64.b64encode(output.getvalue()).decode()

    def preprocess_image(self, image_data: str) -> str:
        """预处理图片以提高OCR质量"""
        try:
            # 解码base64图片数据
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # 增强图像质量
            image = self.enhance_for_ocr(image)

            # 压缩图像到限制大小
            processed_data = self.compress_to_limit(image)

            original_size = len(image_data)
            processed_size = len(processed_data)
            logger.debug(f"Image processed from {original_size} to {processed_size} bytes")

            return processed_data
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理图像OCR请求"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires image_path")

        try:
            # 读取并处理本地图片
            with open(input_data.image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            logger.debug(f"Loaded local image from {input_data.image_path}")

            # 预处理图片
            processed_image = self.preprocess_image(image_data)

            # 准备系统提示词
            system_prompt = (
                "You are an advanced OCR system specialized in extracting text from images. "
                "Instructions:\n"
                "1. Extract ALL text from the image, preserving the original formatting when possible\n"
                "2. Pay special attention to numbers, dates, and tabular data\n"
                "3. If the text is arranged in columns or tables, maintain that structure\n"
                "4. Return the raw text without any additional commentary\n"
                "5. If no text can be found, return 'NO_TEXT_FOUND'\n"
            )
            
            if input_data.question:
                system_prompt += (
                    "6. Focus on text that is relevant to answering the following question: "
                    f"{input_data.question}\n"
                )

            # 使用OpenAI Vision API
            logger.debug("Sending request to OpenAI Vision API")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image, maintaining the original structure and formatting:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{processed_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=128,
                temperature=0
            )

            # 从回复中提取文本
            extracted_text = response.choices[0].message.content.strip()
            logger.info("Successfully extracted text from image")

            # 计算置信度分数
            confidence = 0.9 if extracted_text and extracted_text != "NO_TEXT_FOUND" else 0.1

            return AgentOutput(
                result=extracted_text,
                confidence=confidence,
                metadata={
                    "raw_response": response.model_dump(),
                    "token_usage": response.usage.total_tokens if response.usage else None
                }
            )
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise Exception(f"OCR processing failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.image_path) 