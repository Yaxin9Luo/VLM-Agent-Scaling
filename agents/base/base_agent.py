from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger

@dataclass
class AgentInput:
    """Agent input data structure"""
    image_path: str
    question: str = None
    previous_responses: List['AgentOutput'] = None
    metadata: Dict[str, Any] = None

@dataclass
class AgentOutput:
    """Agent output data structure"""
    result: str
    confidence: float
    metadata: Dict[str, Any] = None

class BaseAgent:
    """Base agent class with collaboration capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.collaboration_manager = None
        
    def set_collaboration_manager(self, manager):
        """Set the collaboration manager for this agent"""
        self.collaboration_manager = manager
        
    def request_collaboration(self, input_data: AgentInput, current_result: AgentOutput) -> AgentOutput:
        """Request collaboration from other agents when needed"""
        if self.collaboration_manager:
            return self.collaboration_manager.request_collaboration(
                self.__class__.__name__.lower().replace('agent', ''),
                input_data,
                current_result
            )
        return current_result
        
    def process(self, input_data: AgentInput) -> AgentOutput:
        """Process the input and generate output, with collaboration if needed"""
        raise NotImplementedError("Subclasses must implement process method")
        
    def validate_input(self, input_data: AgentInput) -> bool:
        """Validate input data"""
        raise NotImplementedError("Subclasses must implement validate_input method")
        
    def _handle_low_confidence(self, input_data: AgentInput, result: AgentOutput) -> AgentOutput:
        """Handle cases where the agent's confidence is low"""
        if result.confidence < 0.7:  # Threshold for low confidence
            return self.request_collaboration(input_data, result)
        return result 