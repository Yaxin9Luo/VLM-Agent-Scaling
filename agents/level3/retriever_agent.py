import os
from typing import Dict, Any, Optional, List
from PIL import Image
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput
from ..base.vlm_base import VLMBase

class RetrieverAgent(BaseAgent):
    """Retriever Agent实现，使用InternVL2-8B来综合所有之前Agent的输出，生成最终答案"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vlm = VLMBase()
        logger.info("Retriever Agent initialized successfully")

    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理所有之前Agent的输出，生成最终答案"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires previous_responses")

        try:
            # 读取图片
            image = Image.open(input_data.image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 准备系统提示词
            system_prompt = (
                "You are an expert AI response synthesizer. Your task is to analyze the image "
                "and all previous outputs to provide the best possible final answer. Consider:\n"
                "1. Accuracy and factual correctness of all inputs\n"
                "2. Consistency between different perspectives\n"
                "3. Completeness of the final answer\n"
                "4. Relevance to the original question\n"
                "5. Clarity and coherence of presentation\n\n"
                "Provide a final, authoritative answer that combines the most reliable "
                "and relevant information from all sources."
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
                f"Please analyze the image and all previous outputs to provide "
                f"the best possible final answer to the question: {input_data.question}"
            )

            # 使用InternVL2-8B模型进行推理
            logger.debug("Running inference with InternVL2-8B model")
            final_result = self.vlm.process_image_query(image, query, system_prompt)
            logger.info("Successfully generated final answer")

            # 计算最终置信度分数
            base_confidence = sum(resp.confidence for resp in input_data.previous_responses) / len(input_data.previous_responses)
            # 最终答案的置信度略高于平均值，但不超过0.98
            confidence = min(0.98, base_confidence * 1.15)

            return AgentOutput(
                result=final_result,
                confidence=confidence,
                metadata={
                    "model": "internvl2-8b",
                    "base_confidence": base_confidence,
                    "question": input_data.question
                }
            )

        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            raise Exception(f"Final answer generation failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.previous_responses and len(input_data.previous_responses) > 0 and input_data.question and input_data.image_path) 