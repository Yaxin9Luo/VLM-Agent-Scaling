import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput

class RetrieverAgent(BaseAgent):
    """Retriever Agent实现，使用OpenAI GPT-4来整合所有Refiner Agents的输出，并回答问题"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key)
        logger.info("Retriever Agent initialized successfully")

    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理来自Refiner Agents的输出，并生成最终答案"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires previous_responses from refiners and a question")

        try:
            # 准备系统提示词
            system_prompt = (
                "You are an expert question answerer. Your task is to provide a clear, accurate, "
                "and concise answer to the given question based on the refined information from "
                "multiple AI agents. Focus on:\n"
                "1. Directly answering the question\n"
                "2. Using the most reliable information from the refinements\n"
                "3. Resolving any conflicts between different refinements\n"
                "4. Providing evidence or reasoning for your answer\n"
                "5. Being concise while including all necessary details\n\n"
                "Provide a final answer that directly addresses the question using the most "
                "accurate and relevant information available."
            )

            # 准备用户提示词，包含问题和每个refiner的输出
            user_prompt = f"Question: {input_data.question}\n\n"
            
            # 添加每个refiner的输出
            for i, resp in enumerate(input_data.previous_responses, 1):
                user_prompt += f"Refiner {i} (confidence: {resp.confidence:.2f}):\n{resp.result}\n\n"

            user_prompt += (
                f"Based on the above information, please provide a clear and concise answer to the question: "
                f"{input_data.question}\n"
                f"Focus on directly answering the question while using the most reliable information available."
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
                max_tokens=1000
            )

            # 提取最终答案
            final_answer = response.choices[0].message.content.strip()
            logger.info("Successfully generated final answer")

            # 计算最终置信度（基于输入的置信度加权平均）
            weights = [resp.confidence for resp in input_data.previous_responses]
            weighted_confidence = sum(w * c for w, c in zip(weights, weights)) / sum(weights)
            
            # 根据输入置信度的一致性调整最终置信度
            confidence_variance = max(weights) - min(weights)
            if confidence_variance < 0.1:  # 如果输入非常一致，略微提高置信度
                final_confidence = min(0.98, weighted_confidence * 1.1)
            else:  # 如果输入不太一致，略微降低置信度
                final_confidence = weighted_confidence * 0.95

            return AgentOutput(
                result=final_answer,
                confidence=final_confidence,
                metadata={
                    "raw_response": response.model_dump(),
                    "token_usage": response.usage.total_tokens if response.usage else None,
                    "input_confidences": weights,
                    "confidence_variance": confidence_variance,
                    "weighted_confidence": weighted_confidence,
                    "question": input_data.question
                }
            )

        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            raise Exception(f"Answer generation failed: {str(e)}")

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(
            input_data.previous_responses and 
            len(input_data.previous_responses) > 0 and
            all(isinstance(resp, AgentOutput) for resp in input_data.previous_responses) and
            input_data.question
        ) 