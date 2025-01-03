import os
from typing import Dict, Any, Optional
from PIL import Image
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput
from ..base.vlm_base import VLMBase

class RelationAgent(BaseAgent):
    """Relation Agent实现，使用InternVL2-8B模型分析图像中的关系信息"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vlm = VLMBase()
        logger.info("Relation Agent initialized successfully")

    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理图像关系分析请求"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires image_path")

        try:
            # 读取图片
            image = Image.open(input_data.image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
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

            # 使用InternVL2-8B模型进行推理
            logger.debug("Running inference with InternVL2-8B model")
            query = "Analyze all relationships present in this image:"
            analysis_result = self.vlm.process_image_query(image, query, system_prompt)
            logger.info("Successfully analyzed relationships in image")

            # 计算置信度分数
            confidence = 0.9 if analysis_result and analysis_result != "NO_RELATIONSHIPS_FOUND" else 0.1

            return AgentOutput(
                result=analysis_result,
                confidence=confidence,
                metadata={
                    "model": "internvl2-8b",
                }
            )
        except Exception as e:
            logger.error(f"Error analyzing image relationships: {str(e)}")
            raise Exception(f"Relationship analysis failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.image_path) 