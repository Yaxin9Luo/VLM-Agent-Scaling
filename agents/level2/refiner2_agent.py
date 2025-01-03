import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput

class Refiner2Agent(BaseAgent):
    """Refiner2 Agent实现，使用OpenAI GPT-4来优化和改进第一层Agent的输出"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key)
        logger.info("Refiner2 Agent initialized successfully")

    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理来自第一层Agent的输出，并进行优化"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires previous_responses and a question")

        try:
            # 准备系统提示词
            system_prompt = (
                "You are an expert AI output synthesizer. Your task is to analyze outputs from "
                "multiple AI agents and synthesize them into a coherent answer to the given question. "
                "Focus on:\n"
                "1. Identifying and removing redundant information\n"
                "2. Combining complementary insights\n"
                "3. Ensuring all information is relevant to the question\n"
                "4. Creating a clear and logical flow of information\n"
                "5. Maintaining accuracy while improving clarity\n\n"
                "Provide a synthesized response that effectively combines the most relevant "
                "information to answer the question."
            )

            # 准备用户提示词
            user_prompt = (
                f"Question: {input_data.question}\n\n"
                f"Previous agents provided the following outputs:\n\n"
            )

            # 添加每个agent的输出
            for i, resp in enumerate(input_data.previous_responses, 1):
                user_prompt += f"Agent {i} output (confidence: {resp.confidence:.2f}):\n{resp.result}\n\n"

            user_prompt += (
                f"Please synthesize these outputs into a coherent response that helps answer "
                f"the question: {input_data.question}\n"
                f"Focus on combining complementary information while removing redundancy."
            )

            # 调用OpenAI API
            logger.debug("Sending request to OpenAI API")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # 使用较低的temperature以保持输出的一致性
                max_tokens=128
            )

            # 提取优化后的结果
            synthesized_result = response.choices[0].message.content.strip()
            logger.info("Successfully synthesized previous outputs")

            # 计算置信度分数（基于输入的置信度和一致性）
            confidences = [resp.confidence for resp in input_data.previous_responses]
            base_confidence = sum(confidences) / len(confidences)
            
            # 根据输入的一致性调整置信度
            confidence_variance = max(confidences) - min(confidences)
            if confidence_variance < 0.2:  # 如果输入比较一致
                confidence = min(0.95, base_confidence * 1.1)
            else:  # 如果输入不太一致
                confidence = base_confidence * 0.9

            return AgentOutput(
                result=synthesized_result,
                confidence=confidence,
                metadata={
                    "raw_response": response.model_dump(),
                    "token_usage": response.usage.total_tokens if response.usage else None,
                    "base_confidence": base_confidence,
                    "confidence_variance": confidence_variance,
                    "question": input_data.question
                }
            )

        except Exception as e:
            logger.error(f"Error synthesizing outputs: {str(e)}")
            raise Exception(f"Synthesis process failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.previous_responses and len(input_data.previous_responses) > 0 and input_data.question) 