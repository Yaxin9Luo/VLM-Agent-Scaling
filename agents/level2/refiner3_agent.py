import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput

class Refiner3Agent(BaseAgent):
    """Refiner3 Agent实现，使用OpenAI GPT-4来优化和改进第一层Agent的输出"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key)
        logger.info("Refiner3 Agent initialized successfully")

    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理来自第一层Agent的输出，并进行优化"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires previous_responses and a question")

        try:
            # 准备系统提示词
            system_prompt = (
                "You are an expert AI output enhancer. Your task is to analyze outputs from "
                "multiple AI agents and enhance them to better answer the given question. "
                "Focus on:\n"
                "1. Identifying gaps in information needed to answer the question\n"
                "2. Adding relevant context and background information\n"
                "3. Making implicit connections explicit\n"
                "4. Providing additional details that strengthen the answer\n"
                "5. Ensuring all enhancements are reliable and supported by the inputs\n\n"
                "Provide an enhanced response that fills information gaps and provides a more "
                "complete answer to the question."
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
                f"Please enhance these outputs to provide a more complete answer to "
                f"the question: {input_data.question}\n"
                f"Focus on filling information gaps and making connections clearer."
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
            enhanced_result = response.choices[0].message.content.strip()
            logger.info("Successfully enhanced previous outputs")

            # 计算置信度分数（基于输入的置信度和信息完整性）
            confidences = [resp.confidence for resp in input_data.previous_responses]
            base_confidence = sum(confidences) / len(confidences)
            
            # 根据输入的可靠性调整置信度
            min_confidence = min(confidences)
            if min_confidence > 0.7:  # 如果所有输入都比较可靠
                confidence = min(0.95, base_confidence * 1.1)
            else:  # 如果有不太可靠的输入
                confidence = base_confidence * 0.9

            return AgentOutput(
                result=enhanced_result,
                confidence=confidence,
                metadata={
                    "raw_response": response.model_dump(),
                    "token_usage": response.usage.total_tokens if response.usage else None,
                    "base_confidence": base_confidence,
                    "min_input_confidence": min_confidence,
                    "question": input_data.question
                }
            )

        except Exception as e:
            logger.error(f"Error enhancing outputs: {str(e)}")
            raise Exception(f"Enhancement process failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.previous_responses and len(input_data.previous_responses) > 0 and input_data.question) 