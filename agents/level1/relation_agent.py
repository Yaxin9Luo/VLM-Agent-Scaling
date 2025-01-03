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
                "You are a specialized Relation Agent focusing on analyzing and describing the relationships between entities (objects, people, scenes) in images or videos.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. IDENTIFY ENTITIES: List out all prominent objects, persons, or elements.\n"
                "2. RELATIONSHIP DETECTION: Describe spatial, logical, or semantic relationships (e.g., 'A is on top of B', 'X is speaking to Y').\n"
                "3. CONTEXT AWARENESS: Understand scenes, contexts, or interactions (e.g., group activities, person-object usage).\n"
                "4. PRECISION: Avoid irrelevant details or speculation beyond the visible or explicitly stated context.\n"
                "5. NO TEXT RECOGNITION: If text is present, do not attempt to read it; that is handled by the OCR Agent.\n"
                "6. KEEP IT STRUCTURED: Provide clear and concise relationship statements.\n"
                "7. NO UNNECESSARY JUDGMENTS: Only report relationships you can justify from the visual data.\n\n"
                "CHAIN OF THOUGHT / REFLECTION / RETHINKING:\n"
                " - Think step by step: identify each entity, then consider all possible pairwise or group relationships.\n"
                " - If a relationship is unclear, reflect on the visual cues. Re-check the scene to confirm.\n"
                " - Provide a final structured description of relationships.\n\n"
                "IMPORTANT: Focus strictly on describing relationships without adding extra info unrelated to entity interactions.\n"
                "Do NOT repeat the user question or instructions in your final output.\n"
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