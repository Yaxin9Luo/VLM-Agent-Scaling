import os
import pytest
from unittest.mock import Mock, patch, mock_open
from PIL import Image
import io
import base64
from agents.level1.relation_agent import RelationAgent
from agents.base.base_agent import AgentInput, AgentOutput

@pytest.fixture
def relation_agent():
    """创建RelationAgent实例的fixture"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        agent = RelationAgent()
        return agent

@pytest.fixture
def sample_image():
    """创建一个简单的测试图片"""
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()

def test_relation_agent_initialization(relation_agent):
    """测试RelationAgent的初始化"""
    assert isinstance(relation_agent, RelationAgent)
    assert relation_agent.api_key == 'test_key'

def test_compress_image(relation_agent, sample_image):
    """测试图片压缩功能"""
    compressed = relation_agent.compress_image(sample_image)
    assert isinstance(compressed, str)
    # 确保可以解码回图片
    img_data = base64.b64decode(compressed)
    img = Image.open(io.BytesIO(img_data))
    assert isinstance(img, Image.Image)

def test_validate_input(relation_agent):
    """测试输入验证"""
    # 测试有效输入
    valid_input_path = AgentInput(image_path="test.jpg")
    assert relation_agent.validate_input(valid_input_path) is True

    valid_input_url = AgentInput(image_url="http://example.com/image.jpg")
    assert relation_agent.validate_input(valid_input_url) is True

    # 测试无效输入
    invalid_input = AgentInput()
    assert relation_agent.validate_input(invalid_input) is False

@pytest.mark.asyncio
async def test_process_with_mock_api(relation_agent, sample_image):
    """测试处理流程（使用mock的API响应）"""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="A cat is sitting on a table"))]
    mock_response.usage = Mock(total_tokens=100)
    mock_response.model_dump = Mock(return_value={"mock": "response"})

    with patch.object(relation_agent.client.chat.completions, 'create', return_value=mock_response):
        input_data = AgentInput(image_path="test.jpg")
        
        # 使用mock_open正确的方式
        mock_file = mock_open()
        mock_file.return_value.read.return_value = base64.b64decode(sample_image)
        
        with patch('builtins.open', mock_file):
            result = await relation_agent.process(input_data)
            
            assert isinstance(result, AgentOutput)
            assert result.result == "A cat is sitting on a table"
            assert result.confidence == 0.9
            assert result.metadata["token_usage"] == 100

def test_invalid_api_key():
    """测试无效的API key"""
    with patch.dict('os.environ', clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found in environment variables"):
            RelationAgent()

@pytest.mark.asyncio
async def test_process_invalid_input(relation_agent):
    """测试处理无效输入"""
    with pytest.raises(ValueError, match="Invalid input: requires either image_path or image_url"):
        await relation_agent.process(AgentInput()) 