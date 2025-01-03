import os
import json
from tqdm import tqdm
from loguru import logger
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

class MMVETTester:
    def __init__(self):
        # 初始化模型
        self.model_path = "/root/autodl-tmp/InternVL2-8B"
        logger.info(f"Initializing InternVL2-8B model from {self.model_path}")
        self.model = pipeline(
            self.model_path,
            backend_config=TurbomindEngineConfig(session_len=8192)
        )
        
        # 设置数据路径
        self.image_dir = "/root/autodl-tmp/VLM-Agent-Scaling/images/mm-vet/images"
        self.results_dir = "/root/autodl-tmp/VLM-Agent-Scaling/results"
        self.mmvet_json = "/root/autodl-tmp/VLM-Agent-Scaling/images/mm-vet/mm-vet.json"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 加载MM-VET问题
        with open(self.mmvet_json, 'r', encoding='utf-8') as f:
            self.mmvet_data = json.load(f)
        
        logger.info("MMVETTester initialized successfully")
    
    def process_single_image(self, image_path: str, question: str) -> str:
        """处理单个图像的问题"""
        try:
            # 加载并处理图像
            image = load_image(image_path)
            
            # 构建提示词
            prompt = (
                "You are an AI assistant specialized in understanding and analyzing images. "
                "Please answer the following question about the image carefully and accurately. "
                "If the question involves calculations, show your work step by step. "
                "If you're not sure about something, say so explicitly.\n\n"
                f"Question: {question}\n"
                "Answer: "
            )
            
            # 获取模型响应
            response = self.model((prompt, image))
            
            # 从Response对象中提取文本
            if hasattr(response, 'text'):
                return response.text.strip()
            else:
                return str(response).strip()
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_test(self):
        """运行完整的MM-VET测试"""
        results = {}
        
        # 处理每个问题
        for image_id, data in tqdm(self.mmvet_data.items(), desc="Processing MM-VET questions"):
            # 获取图片路径和问题
            image_path = os.path.join(self.image_dir, data['imagename'])
            question = data['question']
            
            # 确保图片存在
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                continue
            
            # 处理图像并获取响应
            response = self.process_single_image(image_path, question)
            
            # 保存结果
            results[image_id] = response
            
            # 实时保存结果
            with open(os.path.join(self.results_dir, "mmvet_results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {image_id}")
            logger.info(f"Image: {data['imagename']}")
            logger.info(f"Question: {question}")
            logger.info(f"Response: {response}\n")
        
        logger.info("Testing completed. Results saved to mmvet_results.json")
        return results

if __name__ == "__main__":
    # 设置日志级别
    logger.add("mmvet_test.log", rotation="500 MB")
    
    # 运行测试
    tester = MMVETTester()
    results = tester.run_test() 