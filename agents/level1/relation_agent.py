import os
import base64
import io
from typing import Dict, Any, Optional
from openai import OpenAI
from PIL import Image
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput

class RelationAgent(BaseAgent):
    """Relation Agent实现，使用OpenAI GPT-4 Vision API分析图像中的关系信息"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key)
        logger.info("Relation Agent initialized successfully")

    def compress_image(self, image_data: str, max_bytes: int = 20000000) -> str:
        """压缩图像到API限制大小"""
        try:
            # 解码base64图片数据
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # 如果不是RGB模式，转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 尝试不同的尺寸和质量组合
            sizes = [2048, 1600, 1200, 800]
            qualities = [95, 85, 75, 65]
            
            for size in sizes:
                ratio = size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                resized = image.resize(new_size, Image.Resampling.LANCZOS)
                
                for quality in qualities:
                    output = io.BytesIO()
                    resized.save(output, format='JPEG', quality=quality, optimize=True)
                    data = output.getvalue()
                    
                    if len(data) <= max_bytes:
                        return base64.b64encode(data).decode()
            
            # 如果还是太大，使用最小的尺寸和质量
            output = io.BytesIO()
            final_size = (800, int(800 * image.size[1] / image.size[0]))
            final_image = image.resize(final_size, Image.Resampling.LANCZOS)
            final_image.save(output, format='JPEG', quality=65, optimize=True)
            return base64.b64encode(output.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Error compressing image: {str(e)}")
            raise

    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理图像关系分析请求"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires image_path")

        try:
            # 读取并处理本地图片
            with open(input_data.image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            logger.debug(f"Loaded local image from {input_data.image_path}")

            # 压缩图片
            processed_image = self.compress_image(image_data)

            # 准备系统提示词
            system_prompt = (
                "You are an advanced visual relationship analyzer. "
                "Instructions:\n"
                "1. Analyze and describe all relationships between objects/entities in the image\n"
                "2. Focus on spatial relationships (above, below, next to, etc.)\n"
                "3. Identify interactions between objects/people\n"
                "4. Note any hierarchical or functional relationships\n"
                "5. Be precise and specific in relationship descriptions\n"
                "6. If no relationships can be found, return 'NO_RELATIONSHIPS_FOUND'\n"
            )
            
            if input_data.question:
                system_prompt += (
                    "7. Focus on relationships that are relevant to answering the following question: "
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
                                "text": "Analyze all relationships present in this image:"
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
                max_tokens=1000,
                temperature=0
            )

            # 从回复中提取分析结果
            analysis_result = response.choices[0].message.content.strip()
            logger.info("Successfully analyzed relationships in image")

            # 计算置信度分数
            confidence = 0.9 if analysis_result and analysis_result != "NO_RELATIONSHIPS_FOUND" else 0.1

            return AgentOutput(
                result=analysis_result,
                confidence=confidence,
                metadata={
                    "raw_response": response.model_dump(),
                    "token_usage": response.usage.total_tokens if response.usage else None
                }
            )
        except Exception as e:
            logger.error(f"Error analyzing image relationships: {str(e)}")
            raise Exception(f"Relationship analysis failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.image_path) 