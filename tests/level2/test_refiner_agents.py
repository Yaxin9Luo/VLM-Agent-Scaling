import os
import asyncio
import pytest
from unittest.mock import Mock, patch, mock_open
from PIL import Image
import io
import base64
from agents.level1.ocr_agent import OCRAgent
from agents.level1.relation_agent import RelationAgent
from agents.level1.common_agent import CommonAgent
from agents.level2.refiner1_agent import Refiner1Agent
from agents.level2.refiner2_agent import Refiner2Agent
from agents.level2.refiner3_agent import Refiner3Agent
from agents.base.base_agent import AgentInput, AgentOutput

@pytest.fixture
def sample_image():
    """创建一个简单的测试图片"""
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()

@pytest.fixture
def mock_level1_responses():
    """模拟来自level1 agents的响应"""
    return [
        AgentOutput(
            result="Text detected: 'Hello World'",
            confidence=0.8,
            metadata={"source": "ocr"}
        ),
        AgentOutput(
            result="The text is located at the center of the image",
            confidence=0.85,
            metadata={"source": "relation"}
        ),
        AgentOutput(
            result="A white background with black text",
            confidence=0.9,
            metadata={"source": "common"}
        )
    ]

@pytest.fixture
def refiner1_agent():
    """创建Refiner1Agent实例"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        return Refiner1Agent()

@pytest.fixture
def refiner2_agent():
    """创建Refiner2Agent实例"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        return Refiner2Agent()

@pytest.fixture
def refiner3_agent():
    """创建Refiner3Agent实例"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        return Refiner3Agent()

@pytest.mark.asyncio
async def test_refiner1_process(refiner1_agent, mock_level1_responses):
    """测试Refiner1Agent的处理流程"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Refined output: Clear 'Hello World' text centered on white background"))]
    mock_response.usage = Mock(total_tokens=100)
    mock_response.model_dump = Mock(return_value={"mock": "response"})

    with patch.object(refiner1_agent.client.chat.completions, 'create', return_value=mock_response):
        input_data = AgentInput(previous_responses=mock_level1_responses)
        result = await refiner1_agent.process(input_data)
        
        assert isinstance(result, AgentOutput)
        assert "Refined output" in result.result
        assert 0 < result.confidence <= 0.95
        assert result.metadata["base_confidence"] == sum(r.confidence for r in mock_level1_responses) / len(mock_level1_responses)

@pytest.mark.asyncio
async def test_refiner2_process(refiner2_agent, mock_level1_responses):
    """测试Refiner2Agent的处理流程"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Synthesized: A white background image containing 'Hello World' text in the center"))]
    mock_response.usage = Mock(total_tokens=100)
    mock_response.model_dump = Mock(return_value={"mock": "response"})

    with patch.object(refiner2_agent.client.chat.completions, 'create', return_value=mock_response):
        input_data = AgentInput(previous_responses=mock_level1_responses)
        result = await refiner2_agent.process(input_data)
        
        assert isinstance(result, AgentOutput)
        assert "Synthesized" in result.result
        assert 0 < result.confidence <= 0.95
        assert "confidence_variance" in result.metadata

@pytest.mark.asyncio
async def test_refiner3_process(refiner3_agent, mock_level1_responses):
    """测试Refiner3Agent的处理流程"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Enhanced: The image shows 'Hello World' text prominently displayed in black font against a clean white background, suggesting a minimalist design approach"))]
    mock_response.usage = Mock(total_tokens=100)
    mock_response.model_dump = Mock(return_value={"mock": "response"})

    with patch.object(refiner3_agent.client.chat.completions, 'create', return_value=mock_response):
        input_data = AgentInput(previous_responses=mock_level1_responses)
        result = await refiner3_agent.process(input_data)
        
        assert isinstance(result, AgentOutput)
        assert "Enhanced" in result.result
        assert 0 < result.confidence <= 0.9  # Refiner3的最大置信度是0.9
        assert "enhancement_factor" in result.metadata

@pytest.mark.asyncio
async def test_integration_with_level1(sample_image, refiner1_agent, refiner2_agent, refiner3_agent):
    """测试与level1 agents的集成"""
    # 模拟level1 agents的响应
    mock_ocr_response = Mock()
    mock_ocr_response.choices = [Mock(message=Mock(content="Text: Hello World"))]
    mock_ocr_response.usage = Mock(total_tokens=50)
    mock_ocr_response.model_dump = Mock(return_value={"mock": "response"})

    mock_relation_response = Mock()
    mock_relation_response.choices = [Mock(message=Mock(content="Centered text"))]
    mock_relation_response.usage = Mock(total_tokens=50)
    mock_relation_response.model_dump = Mock(return_value={"mock": "response"})

    mock_common_response = Mock()
    mock_common_response.choices = [Mock(message=Mock(content="White background with text"))]
    mock_common_response.usage = Mock(total_tokens=50)
    mock_common_response.model_dump = Mock(return_value={"mock": "response"})

    # 模拟refiners的响应
    mock_refiner1_response = Mock()
    mock_refiner1_response.choices = [Mock(message=Mock(content="Refined: Clear text on white"))]
    mock_refiner1_response.usage = Mock(total_tokens=100)
    mock_refiner1_response.model_dump = Mock(return_value={"mock": "response"})

    mock_refiner2_response = Mock()
    mock_refiner2_response.choices = [Mock(message=Mock(content="Synthesized: Minimal design"))]
    mock_refiner2_response.usage = Mock(total_tokens=100)
    mock_refiner2_response.model_dump = Mock(return_value={"mock": "response"})

    mock_refiner3_response = Mock()
    mock_refiner3_response.choices = [Mock(message=Mock(content="Enhanced: Professional layout"))]
    mock_refiner3_response.usage = Mock(total_tokens=100)
    mock_refiner3_response.model_dump = Mock(return_value={"mock": "response"})

    # 设置所有需要的mock
    mock_file = mock_open()
    mock_file.return_value.read.return_value = base64.b64decode(sample_image)
    
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}), \
         patch('builtins.open', mock_file), \
         patch('openai.resources.chat.completions.Completions.create',
               side_effect=[
                   mock_ocr_response,
                   mock_relation_response,
                   mock_common_response,
                   mock_refiner1_response,
                   mock_refiner2_response,
                   mock_refiner3_response
               ]):
        
        # 运行level1 agents
        ocr_agent = OCRAgent()
        relation_agent = RelationAgent()
        common_agent = CommonAgent()

        input_data = AgentInput(image_path="test.jpg")
        level1_results = await asyncio.gather(
            ocr_agent.process(input_data),
            relation_agent.process(input_data),
            common_agent.process(input_data)
        )

        # 运行refiners
        refiner_input = AgentInput(previous_responses=level1_results)
        refiner_results = await asyncio.gather(
            refiner1_agent.process(refiner_input),
            refiner2_agent.process(refiner_input),
            refiner3_agent.process(refiner_input)
        )

        # 验证结果
        assert len(refiner_results) == 3
        for result in refiner_results:
            assert isinstance(result, AgentOutput)
            assert result.result
            assert 0 < result.confidence <= 1
            assert result.metadata

@pytest.mark.asyncio
async def test_invalid_input_refiner1(refiner1_agent):
    """测试Refiner1Agent的无效输入处理"""
    with pytest.raises(ValueError, match="Invalid input: requires previous_responses"):
        await refiner1_agent.process(AgentInput())

@pytest.mark.asyncio
async def test_invalid_input_refiner2(refiner2_agent):
    """测试Refiner2Agent的无效输入处理"""
    with pytest.raises(ValueError, match="Invalid input: requires previous_responses"):
        await refiner2_agent.process(AgentInput())

@pytest.mark.asyncio
async def test_invalid_input_refiner3(refiner3_agent):
    """测试Refiner3Agent的无效输入处理"""
    with pytest.raises(ValueError, match="Invalid input: requires previous_responses"):
        await refiner3_agent.process(AgentInput()) 