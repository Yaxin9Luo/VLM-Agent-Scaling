import os
from typing import Dict, Any, Optional
from PIL import Image
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput
from ..base.vlm_base import VLMBase

class CommonAgent(BaseAgent):
    """Common Agent实现，使用InternVL2-8B模型处理通用视觉问答任务"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vlm = VLMBase()
        logger.info("Common Agent initialized successfully")

    def process(self, input_data: AgentInput) -> AgentOutput:
        """处理通用视觉问答请求"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires image_path and question")

        try:
            # 读取图片
            image = Image.open(input_data.image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 准备系统提示词
            system_prompt = (
                "You are the Common Agent designed to handle general queries about images or videos. "
                "You address tasks that are not explicitly covered by the specialized OCR Agent or Relation Agent.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. MULTI-PURPOSE ANALYSIS: Handle color, shape, context, scene classification, object recognition, and other general aspects.\n"
                "2. VISUAL DETAILS: Provide relevant descriptions about how things look without reading text or focusing on explicit relationships.\n"
                "3. AVOID DUPLICATION: If the question pertains to text reading or specific relational inference, defer to those specialized agents.\n"
                "4. NO HALLUCINATION: Only describe what can be directly observed or reasonably inferred.\n"
                "5. STRUCTURE: Provide answers in a concise, factual manner.\n\n"
                "CHAIN OF THOUGHT / REFLECTION / RETHINKING:\n"
                " - Carefully consider the entire image or video context to identify general features.\n"
                " - If uncertain, reflect on whether a specialized Agent (OCR or Relation) would handle this better.\n"
                " - Provide a final factual overview.\n\n"
                "IMPORTANT: Your focus is broad visual analysis without stepping into text reading or complex relationship extraction.\n"
                "Do NOT repeat the user question or instructions in your final output.\n"
            )


            # 使用InternVL2-8B模型进行推理
            logger.debug("Running inference with InternVL2-8B model")
            query = input_data.question if input_data.question else "Describe this image in detail:"
            answer = self.vlm.process_image_query(image, query, system_prompt)
            logger.info("Successfully processed visual question")

            # 计算置信度分数
            confidence = 0.9 if answer and answer != "CANNOT_ANSWER" else 0.1

            return AgentOutput(
                result=answer,
                confidence=confidence,
                metadata={
                    "model": "internvl2-8b",
                }
            )
        except Exception as e:
            logger.error(f"Error processing visual question: {str(e)}")
            raise Exception(f"Visual question answering failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.image_path) and bool(input_data.question) 