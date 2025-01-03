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

            system_prompt = (
                "You are a Refiner Agent. Your task is to refine and enhance the responses provided by the first-layer Agents (OCR, Relation, Common). "
                "You do NOT look at the raw image or video directly; instead, you improve the existing answers based on logic, consistency, and clarity.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. GATHER RESPONSES: Take the answers from the specialized Agents as input.\n"
                "2. VERIFY ACCURACY: Check if the combined content is logically consistent and answers the given questions.\n"
                "3. REMOVE REDUNDANCY: Eliminate duplicated statements or irrelevant details.\n"
                "4. FILL GAPS: If the specialized Agents' answers suggest missing information or potential confusion, clarify or refine it logically.\n"
                "5. IMPROVE CLARITY: Make the language more concise, direct, and coherent.\n"
                "6. NO SPECULATION: Do not invent new facts beyond what the specialized Agents have stated.\n\n"
                "CHAIN OF THOUGHT / REFLECTION / RETHINKING:\n"
                " - Step through each provided answer carefully, noting any contradictions or unclear points.\n"
                " - Reflect on how to unify them into a single refined response.\n"
                " - Rethink if any improvements to readability or logical flow are needed before finalizing.\n\n"
                "IMPORTANT: Your main goal is to refine. Do not contradict validated facts or add unsupported information.\n"
                "Do NOT repeat the user question or instructions in your final output.\n"
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