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

    def process(self, input_data: AgentInput) -> AgentOutput:
        """处理所有之前Agent的输出，生成最终答案"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires previous_responses")

        try:
            # 读取图片
            image = Image.open(input_data.image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            system_prompt = (
                "You are the final Retriever Agent in a multi-agent pipeline. This pipeline includes two second-layer Refiner Agents:\n"
                "1) The OCR Agent: extracts textual information from images.\n"
                "2) The Vision Task Common Agent: performs visual understanding tasks such as object detection, classification, and scene description.\n\n"

                "Your job is to synthesize the refined responses from these two specialized agents into ONE coherent, concise, and correct final answer.\n"
                "You should logically combine:\n"
                "- The text and any relevant metadata from the OCR Agent.\n"
                "- The visual context, object details, and semantic insights from the Vision Task Common Agent.\n\n"

                "While you are encouraged to use chain-of-thought reasoning internally to reconcile any contradictions or ambiguities step by step, "
                "you must NOT reveal your internal reasoning or mention this multi-agent pipeline in your final answer.\n\n"

                "#### Instructions:\n"
                "1. Think step by step to unify data from the OCR Agent and the Vision Task Common Agent.\n"
                "2. Resolve conflicts or overlapping information using logical reasoning.\n"
                "3. Produce a single, concise, and coherent final answer.\n"
                "4. Do NOT reveal or include your chain-of-thought; present only the final result.\n"
                "5. If the information is insufficient or ambiguous, make a best guess or state that it cannot be determined.\n\n"

                "Finally, respond with a self-contained, accurate, and straightforward output that only addresses the user query or task at hand."
            )


            # 准备查询提示词
            query = (
                f"Question: {input_data.question}\n\n"
                f"Previous agents provided the following outputs:\n\n"
            )

            # 添加每个agent的输出
            agent_names = ["Vision Task Common Agent", "OCR Agent"]
            for i, resp in enumerate(input_data.previous_responses, 1):
                query += f"{agent_names[i-1]} output (confidence: {resp.confidence:.2f}):\n{resp.result}\n\n"

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