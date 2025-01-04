import os
from typing import Dict, Any, Optional
from PIL import Image
from loguru import logger
from ..base.base_agent import BaseAgent, AgentInput, AgentOutput
from ..base.vlm_base import VLMBase

class CommonAgent(BaseAgent):
    """Common Agent实现,使用InternVL2-8B模型处理通用视觉问答任务"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vlm = VLMBase()
        logger.info("Common Agent initialized successfully")

    def process(self, input_data: AgentInput) -> AgentOutput:
        """处理通用视觉问答请求"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: requires image_path and question")
        
        # 1) Parse image and ensure it is in RGB
        try:
            image = Image.open(input_data.image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise
        
        # 2) Gather OCR text from previous responses, if any
        ocr_text = ""
        if input_data.previous_responses:
            for resp in input_data.previous_responses:
                if isinstance(resp, AgentOutput) and resp.metadata.get("agent_name") == "ocr":
                    # 'resp.result' should contain the recognized text from OCR
                    ocr_text = resp.result  
        
        # 3) Build the system prompt by injecting recognized text
        # system_prompt = (
        #     "You are an AI assistant specialized in understanding and analyzing images. "            
        #     "Please answer the following question about the image carefully and accurately. "
        #     "If the question involves calculations, show your reasoning step by step. "
        #     "If you're unsure, say so explicitly.\n\n"
        # )
        system_prompt = (
            "You are an AI assistant specialized in understanding and analyzing images. "
            "If there is any text in the image, you have access to the OCR results. "
            "Below is the recognized text (if any) from the image:\n\n"
            f"{ocr_text}\n\n"
            "Please answer the following question about the image carefully and accurately. "
            "If the question involves calculations, show your reasoning step by step. "
            "If you're unsure, say so explicitly.\n\n"
        )
            
        #     "## Reference Examples:\n"
        #     "1) When the user asks: 'What is written in the middle of the sign?' \n"
        #     "   - Suppose the OCR text is: 'WELCOME TO THE PARK' \n"
        #     "   - A good answer might be: 'It says “WELCOME TO THE PARK” in large letters.' \n"
        #     "   - Explanation: The text is directly from the OCR. We mention its location (middle of the sign) "
        #     "     and confirm that’s exactly what’s written.\n\n"
            
        #     "2) When the user asks: 'How many columns of numbers are shown?' \n"
        #     "   - Suppose the recognized text is:\n"
        #     "         4 + 7 =  2 + 2 =\n"
        #     "         7 + 2 =  6 + 1 =\n"
        #     "   - A good answer might be: 'I see two columns of equations. In the first column: 4+7=, 7+2=, etc. In the second: 2+2=, 6+1=, etc.'\n"
        #     "   - Explanation: We interpret that the text is arranged in two columns of math equations.\n\n"
            
        #     "3) When the user asks: 'Who scored the highest?' \n"
        #     "   - Suppose the recognized text is:\n"
        #     "         Name  | Score\n"
        #     "         Alice | 90\n"
        #     "         Bob   | 85\n"
        #     "   - A good answer might be: 'Alice scored the highest with 90 points.'\n"
        #     "   - Explanation: We use the OCR text to identify a table structure, then answer from that data.\n\n"
            
        #     "Remember: Provide clear, direct answers based on both the visual content and any extracted text. "
        #     "Do not invent details or fabricate text. If you are uncertain, say so. If there's no text provided, "
        #     "focus on purely visual details. If calculations are needed, show your steps.\n"
        # )
        
        # 4) Prepare the actual query
        query = input_data.question or "Describe this image in detail:"
        
        # 5) Forward everything to the VLM
        logger.debug("Running inference with InternVL2-8B model")
        try:
            answer = self.vlm.process_image_query(
                image=image, 
                query=query, 
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"Error processing visual question: {str(e)}")
            raise Exception(f"Visual question answering failed: {str(e)}")
        
        # 6) Compute confidence and return
        confidence = 0.9 if answer and answer != "CANNOT_ANSWER" else 0.1
        return AgentOutput(
            result=answer,
            confidence=confidence,
            metadata={
                "agent_name": "common",
                "model": "internvl2-8b",
                # Possibly store the OCR text again, or any other metadata
                "ocr_text": ocr_text,
            }
        )

    def validate_input(self, input_data: AgentInput) -> bool:
        """验证输入数据"""
        return bool(input_data.image_path) and bool(input_data.question) 