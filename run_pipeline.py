import os
from loguru import logger
from agents.level1.ocr_agent import OCRAgent
from agents.level1.common_agent import CommonAgent
from agents.base.base_agent import AgentInput, AgentOutput
from agents.base.vlm_base import VLMBase
from agents.base.collaboration_manager import CollaborationManager

def run_pipeline(image_path: str, question: str = None):
    """Run the complete Agent Pipeline with collaboration"""
    try:
        # Ensure question parameter exists
        if question is None:
            question = "What can you see in this image?"
        
        # Initialize VLM model (singleton pattern)
        logger.info("Initializing InternVL2-8B model...")
        VLMBase()

        # Initialize collaboration manager and agents
        logger.info("Initializing collaboration manager and agents...")
        collaboration_manager = CollaborationManager()
        
        ocr_agent = OCRAgent()
        common_agent = CommonAgent()
        
        # Register agents with collaboration manager
        collaboration_manager.register_agent("ocr", ocr_agent)
        collaboration_manager.register_agent("common", common_agent)
        
        # Set collaboration manager for each agent
        ocr_agent.set_collaboration_manager(collaboration_manager)
        common_agent.set_collaboration_manager(collaboration_manager)

        # Prepare input data
        input_data = AgentInput(image_path=image_path, question=question)
        
        # Run OCR Agent
        logger.info("Running OCR Agent...")
        ocr_output = ocr_agent.process(input_data)
        logger.info(f"OCR Agent output (confidence: {ocr_output.confidence:.2f}):\n{ocr_output.result}")
        
        # Cross-validate OCR results if confidence is low
        ocr_output = collaboration_manager.cross_validate(ocr_output, input_data, "ocr")
        
        # Run Common Agent with OCR results
        logger.info("Running Common Agent...")
        common_input = AgentInput(
            image_path=image_path,
            question=question,
            previous_responses=[ocr_output]
        )
        common_output = common_agent.process(common_input)
        logger.info(f"Common Agent output (confidence: {common_output.confidence:.2f}):\n{common_output.result}")
        
        # Cross-validate Common Agent results if confidence is low
        final_output = collaboration_manager.cross_validate(common_output, input_data, "common")

        logger.info("Final output from Vision Task Pipeline:")
        logger.info(f"Confidence: {final_output.confidence:.2f}")
        logger.info("Result:")
        logger.info(final_output.result)
        
        return final_output

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Set test image path and question
    image_path = "/root/autodl-tmp/VLM-Agent-Scaling/images/mm-vet/images/v1_1.png"
    question = "Solve the questions"
    run_pipeline(image_path, question) 