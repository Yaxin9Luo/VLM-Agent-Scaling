from typing import Dict, List, Optional, Type
from loguru import logger
from .base_agent import BaseAgent, AgentInput, AgentOutput

class CollaborationManager:
    """Manages collaboration between agents, including feedback loops and cross-validation"""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self.confidence_threshold = 0.7
        self.max_collaboration_rounds = 3
        
    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent with the collaboration manager"""
        self._agents[name] = agent
        logger.info(f"Registered agent: {name}")
        
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name"""
        return self._agents.get(name)
        
    def request_collaboration(self, 
                            requesting_agent: str, 
                            input_data: AgentInput, 
                            current_result: AgentOutput) -> AgentOutput:
        """Handle collaboration requests between agents"""
        if current_result.confidence >= self.confidence_threshold:
            return current_result
            
        logger.info(f"Agent {requesting_agent} requested collaboration due to low confidence: {current_result.confidence}")
        
        if requesting_agent == "ocr":
            # If OCR confidence is low, ask Common agent to verify
            common_agent = self.get_agent("common")
            if common_agent:
                verification_input = AgentInput(
                    image_path=input_data.image_path,
                    question="Can you verify if there is any text in this image? If yes, what is it?",
                    previous_responses=[current_result]
                )
                verification_result = common_agent.process(verification_input)
                
                # Combine results and adjust confidence
                if verification_result.confidence > current_result.confidence:
                    return verification_result
                    
        elif requesting_agent == "common":
            # If Common agent is uncertain, request OCR to focus on specific areas
            ocr_agent = self.get_agent("ocr")
            if ocr_agent:
                # Request detailed OCR analysis
                detailed_input = AgentInput(
                    image_path=input_data.image_path,
                    question="Please perform a detailed OCR analysis, focusing on any text that might help answer: " + input_data.question,
                    previous_responses=[current_result]
                )
                ocr_result = ocr_agent.process(detailed_input)
                
                # If OCR finds relevant text, let Common agent try again
                if ocr_result.confidence > 0.5:
                    new_input = AgentInput(
                        image_path=input_data.image_path,
                        question=input_data.question,
                        previous_responses=[ocr_result, current_result]
                    )
                    return self.get_agent("common").process(new_input)
                    
        return current_result
        
    def cross_validate(self, 
                      primary_result: AgentOutput, 
                      input_data: AgentInput,
                      primary_agent: str) -> AgentOutput:
        """Cross validate results between agents"""
        if primary_result.confidence >= self.confidence_threshold:
            return primary_result
            
        # Get the other agent for validation
        validator_name = "common" if primary_agent == "ocr" else "ocr"
        validator = self.get_agent(validator_name)
        
        if not validator:
            return primary_result
            
        # Perform validation
        validation_input = AgentInput(
            image_path=input_data.image_path,
            question=input_data.question,
            previous_responses=[primary_result]
        )
        validation_result = validator.process(validation_input)
        
        # Compare and merge results
        if validation_result.confidence > primary_result.confidence:
            logger.info(f"Cross-validation: {validator_name} provided better result")
            return validation_result
        
        return primary_result 