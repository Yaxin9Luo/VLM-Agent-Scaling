import asyncio
import os
from loguru import logger
from agents.level1.ocr_agent import OCRAgent
from agents.level1.relation_agent import RelationAgent
from agents.level1.common_agent import CommonAgent
from agents.level2.refiner1_agent import Refiner1Agent
from agents.level2.refiner2_agent import Refiner2Agent
from agents.level2.refiner3_agent import Refiner3Agent
from agents.level3.retriever_agent import RetrieverAgent
from agents.base.base_agent import AgentInput, AgentOutput

async def run_pipeline(image_path: str, question: str = None):
    """运行完整的Agent Pipeline"""
    try:
        # 初始化所有agents
        logger.info("Initializing agents...")
        ocr_agent = OCRAgent()
        relation_agent = RelationAgent()
        common_agent = CommonAgent()
        refiner1 = Refiner1Agent()
        refiner2 = Refiner2Agent()
        refiner3 = Refiner3Agent()
        retriever = RetrieverAgent()

        # Level 1: 运行基础agents
        logger.info("Running Level 1 agents...")
        input_data = AgentInput(image_path=image_path, question=question)
        
        ocr_output = await ocr_agent.process(input_data)
        logger.info(f"OCR Agent output (confidence: {ocr_output.confidence:.2f}):\n{ocr_output.result}")
        
        relation_output = await relation_agent.process(input_data)
        logger.info(f"Relation Agent output (confidence: {relation_output.confidence:.2f}):\n{relation_output.result}")
        
        common_output = await common_agent.process(input_data)
        logger.info(f"Common Agent output (confidence: {common_output.confidence:.2f}):\n{common_output.result}")

        # Level 2: 运行Refiner agents
        logger.info("Running Level 2 agents...")
        level1_outputs = [ocr_output, relation_output, common_output]
        refiner_input = AgentInput(image_path=image_path, question=question, previous_responses=level1_outputs)
        
        refiner1_output = await refiner1.process(refiner_input)
        logger.info(f"Refiner1 output (confidence: {refiner1_output.confidence:.2f}):\n{refiner1_output.result}")
        
        refiner2_output = await refiner2.process(refiner_input)
        logger.info(f"Refiner2 output (confidence: {refiner2_output.confidence:.2f}):\n{refiner2_output.result}")
        
        refiner3_output = await refiner3.process(refiner_input)
        logger.info(f"Refiner3 output (confidence: {refiner3_output.confidence:.2f}):\n{refiner3_output.result}")

        # Level 3: 运行Retriever agent
        logger.info("Running Level 3 agent...")
        refiner_outputs = [refiner1_output, refiner2_output, refiner3_output]
        retriever_input = AgentInput(image_path=image_path, question=question, previous_responses=refiner_outputs)
        
        final_output = await retriever.process(retriever_input)
        logger.info("Final output from Retriever agent:")
        logger.info(f"Confidence: {final_output.confidence:.2f}")
        logger.info("Result:")
        logger.info(final_output.result)
        
        return final_output

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # 检查OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        # 这里需要你输入你的OpenAI API Key
        api_key = input("请输入你的OpenAI API Key: ")
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info("API Key has been set")
    
    image_path = "images/ocr_image.jpg"
    asyncio.run(run_pipeline(image_path)) 