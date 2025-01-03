from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class AgentOutput:
    """Agent的输出结构"""
    result: str
    confidence: float
    metadata: Dict[str, Any] = None

@dataclass
class AgentInput:
    """Agent的输入结构"""
    image_path: Optional[str] = None
    question: Optional[str] = None
    previous_responses: Optional[List[AgentOutput]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseAgent(ABC):
    """基础Agent类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化Agent"""
        self.config = config or {}
    
    @abstractmethod
    async def process(self, input_data: AgentInput) -> AgentOutput:
        """处理输入数据并返回结果"""
        pass
    
    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return True 