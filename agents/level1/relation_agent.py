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

    def process(self, input_data: AgentInput) -> AgentOutput:
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
                "You are an advanced visual relationship analyzer. Your task is to analyze relationships between elements in the image. "
                "Critical Instructions:\n"
                "1. ANALYZE ALL RELATIONSHIPS:\n"
                "   - Spatial relationships (above, below, next to)\n"
                "   - Interactions between objects/people\n"
                "   - Hierarchical relationships\n"
                "   - Functional relationships\n"
                "2. BE SPECIFIC:\n"
                "   - Use precise spatial terms\n"
                "   - Describe exact positions\n"
                "   - Mention specific interactions\n"
                "3. FOCUS ON:\n"
                "   - Object-to-object relationships\n"
                "   - Object-to-environment relationships\n"
                "   - Character/person interactions\n"
                "4. FORMAT:\n"
                "   - List each relationship clearly\n"
                "   - Use simple, direct language\n"
                "5. IMPORTANT: If no relationships can be found, return 'NO_RELATIONS_FOUND'\n"
            )
            
            if input_data.question:
                system_prompt += (
                    "\nWhile analyzing all relationships, pay special attention to those relevant to: "
                    f"{input_data.question}\n"
                )

            # 使用InternVL2-8B模型进行推理
            logger.debug("Running inference with InternVL2-8B model")
            query = (
                "Analyze and list ALL relationships present in this image. "
                "Include spatial, interactive, and functional relationships. "
                "If no relationships exist, respond with 'NO_RELATIONS_FOUND'."
            )
            analysis_result = self.vlm.process_image_query(image, query, system_prompt)
            logger.info("Successfully analyzed relationships in image")

            # 计算置信度分数
            confidence = 0.9 if analysis_result and analysis_result.strip() != "NO_RELATIONS_FOUND" else 0.1

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