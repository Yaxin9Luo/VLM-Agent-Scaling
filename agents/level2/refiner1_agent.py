import os
from typing import Dict, Any, Optional, List
from PIL import Image
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput
from ..base.vlm_base import VLMBase

class Refiner1Agent(BaseAgent):
    """Refiner1 Agent实现，使用InternVL2-8B来优化和改进第一层Agent的输出"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vlm = VLMBase()
        logger.info("Refiner1 Agent initialized successfully")

    def process(self, input_data: AgentInput) -> AgentOutput:
        """处理来自第一层Agent的输出，并进行优化"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires previous_responses")

        try:
            # 读取图片
            image = Image.open(input_data.image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 准备系统提示词
            system_prompt = (
                "You are an expert AI output refiner. Your task is to analyze and improve the outputs "
                "from previous AI agents, focusing specifically on answering the given question. "
                "Consider the following aspects in your refinement:\n"
                "1. Accuracy and factual correctness\n"
                "2. Relevance to the question\n"
                "3. Completeness of information needed to answer the question\n"
                "4. Clarity and coherence\n"
                "5. Consistency across different analyses\n\n"
                "Provide a refined and improved version that focuses on information relevant "
                "to answering the question while maintaining accuracy and completeness."
            )

            # 准备查询提示词
            query = (
                f"Question: {input_data.question}\n\n"
                f"Previous agents provided the following outputs:\n\n"
            )

            # 添加每个agent的输出
            for i, resp in enumerate(input_data.previous_responses, 1):
                query += f"Agent {i} output (confidence: {resp.confidence:.2f}):\n{resp.result}\n\n"

            query += (
                f"Please analyze the image and these outputs, then provide a refined answer "
                f"to the question: {input_data.question}"
            )

            # 使用InternVL2-8B模型进行推理
            logger.debug("Running inference with InternVL2-8B model")
            refined_result = self.vlm.process_image_query(image, query, system_prompt)
            logger.info("Successfully refined previous outputs")

            # 计算置信度分数（基于输入的置信度和优化程度）
            base_confidence = sum(resp.confidence for resp in input_data.previous_responses) / len(input_data.previous_responses)
            confidence = min(0.95, base_confidence * 1.1)  # 略微提高置信度，但不超过0.95

            return AgentOutput(
                result=refined_result,
                confidence=confidence,
                metadata={
                    "model": "internvl2-8b",
                    "base_confidence": base_confidence,
                    "question": input_data.question
                }
            )

        except Exception as e:
            logger.error(f"Error refining outputs: {str(e)}")
            raise Exception(f"Refinement process failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.previous_responses and len(input_data.previous_responses) > 0 and input_data.question and input_data.image_path) 