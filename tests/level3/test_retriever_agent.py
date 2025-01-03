import os
import pytest
from unittest.mock import Mock, patch
from agents.level3.retriever_agent import RetrieverAgent
from agents.base.base_agent import AgentInput, AgentOutput

@pytest.fixture
def mock_refiner_responses():
    """模拟来自refiner agents的响应"""
    return [
        AgentOutput(
            result="Refined: Clear text on white background",
            confidence=0.85,
            metadata={"source": "refiner1"}
        ),
        AgentOutput(
            result="Synthesized: Minimal design with centered text",
            confidence=0.88,
            metadata={"source": "refiner2"}
        ),
        AgentOutput(
            result="Enhanced: Professional layout with emphasis on typography",
            confidence=0.82,
            metadata={"source": "refiner3"}
        )
    ]

@pytest.fixture
def retriever_agent():
    """创建RetrieverAgent实例"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        return RetrieverAgent()

def test_retriever_agent_initialization(retriever_agent):
    """测试RetrieverAgent的初始化"""
    assert isinstance(retriever_agent, RetrieverAgent)
    assert retriever_agent.api_key == 'test_key'

@pytest.mark.asyncio
async def test_process_with_mock_api(retriever_agent, mock_refiner_responses):
    """测试处理流程"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(
        content="Final conclusion: The image features a professionally designed, "
                "minimalist layout with clear text centered on a white background, "
                "demonstrating careful attention to typography and visual hierarchy."
    ))]
    mock_response.usage = Mock(total_tokens=100)
    mock_response.model_dump = Mock(return_value={"mock": "response"})

    with patch.object(retriever_agent.client.chat.completions, 'create', return_value=mock_response):
        input_data = AgentInput(previous_responses=mock_refiner_responses)
        result = await retriever_agent.process(input_data)
        
        assert isinstance(result, AgentOutput)
        assert "Final conclusion" in result.result
        assert 0 < result.confidence <= 0.98
        assert "weighted_confidence" in result.metadata
        assert "confidence_variance" in result.metadata
        assert "input_confidences" in result.metadata

@pytest.mark.asyncio
async def test_process_with_consistent_confidences(retriever_agent):
    """测试处理具有一致置信度的输入"""
    consistent_responses = [
        AgentOutput(result="Output 1", confidence=0.9, metadata={}),
        AgentOutput(result="Output 2", confidence=0.91, metadata={}),
        AgentOutput(result="Output 3", confidence=0.89, metadata={})
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Final output"))]
    mock_response.usage = Mock(total_tokens=100)
    mock_response.model_dump = Mock(return_value={"mock": "response"})

    with patch.object(retriever_agent.client.chat.completions, 'create', return_value=mock_response):
        input_data = AgentInput(previous_responses=consistent_responses)
        result = await retriever_agent.process(input_data)
        
        assert result.confidence > 0.9  # 由于一致性高，置信度应该提高

@pytest.mark.asyncio
async def test_process_with_inconsistent_confidences(retriever_agent):
    """测试处理具有不一致置信度的输入"""
    inconsistent_responses = [
        AgentOutput(result="Output 1", confidence=0.9, metadata={}),
        AgentOutput(result="Output 2", confidence=0.6, metadata={}),
        AgentOutput(result="Output 3", confidence=0.75, metadata={})
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Final output"))]
    mock_response.usage = Mock(total_tokens=100)
    mock_response.model_dump = Mock(return_value={"mock": "response"})

    with patch.object(retriever_agent.client.chat.completions, 'create', return_value=mock_response):
        input_data = AgentInput(previous_responses=inconsistent_responses)
        result = await retriever_agent.process(input_data)
        
        assert result.confidence < 0.9  # 由于一致性低，置信度应该降低

@pytest.mark.asyncio
async def test_validate_input_with_invalid_data(retriever_agent):
    """测试输入验证 - 无效数据"""
    # 测试空输入
    assert not retriever_agent.validate_input(AgentInput())
    
    # 测试空列表
    assert not retriever_agent.validate_input(AgentInput(previous_responses=[]))
    
    # 测试无效类型
    with pytest.raises(ValueError):
        retriever_agent.validate_input(AgentInput(previous_responses=["not an AgentOutput"]))

@pytest.mark.asyncio
async def test_error_handling(retriever_agent, mock_refiner_responses):
    """测试错误处理"""
    with patch.object(retriever_agent.client.chat.completions, 'create', side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="Retrieval process failed: API Error"):
            input_data = AgentInput(previous_responses=mock_refiner_responses)
            await retriever_agent.process(input_data) 