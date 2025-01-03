import os
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv
from run_pipeline import run_pipeline
from agents.base.vlm_base import VLMBase

# 加载环境变量
load_dotenv()

def run_mmvet_benchmark(mmvet_json_path: str, images_dir: str, output_path: str):
    """运行MMVET benchmark测试"""
    try:
        # 初始化VLM模型（单例模式，只会初始化一次）
        logger.info("Initializing InternVL2-8B model...")
        VLMBase()

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 加载MMVET测试集
        with open(mmvet_json_path, 'r', encoding='utf-8') as f:
            mmvet_data = json.load(f)
        
        # 如果结果文件已存在，加载已有结果
        results = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"Loaded {len(results)} existing results")
            except json.JSONDecodeError:
                logger.warning("Failed to load existing results, starting fresh")
        
        # 处理每个测试样例
        for test_id, test_info in tqdm(mmvet_data.items(), desc="Processing MMVET tests"):
            # 如果已经处理过这个测试样例，跳过
            if test_id in results:
                logger.info(f"Skipping {test_id} (already processed)")
                continue
                
            logger.info(f"Processing test: {test_id}")
            
            try:
                # 构建图片完整路径
                image_path = os.path.join(images_dir, test_info['imagename'])
                
                # 准备输入数据
                input_data = {
                    'image_path': image_path,
                    'question': test_info['question'],
                    'capabilities': test_info['capability']
                }
                
                # 运行pipeline
                output = run_pipeline(str(image_path), question=test_info['question'])
                
                # 将结果添加到结果字典
                results[test_id] = output.result
                
                # 立即保存当前结果
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                
                logger.info(f"Completed processing {test_id} with confidence: {output.confidence:.2f}")
                logger.info(f"Question: {test_info['question']}")
                logger.info(f"Answer: {output.result}")
                logger.info("Results saved")
                
            except Exception as e:
                logger.error(f"Error processing {test_id}: {str(e)}")
                # 如果处理失败，添加一个空结果并保存
                results[test_id] = ""
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
        
        logger.info(f"All tests completed. Results saved to {output_path}")
        return results
        
    except Exception as e:
        logger.error(f"Benchmark run failed: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置路径
    mmvet_json_path = "images/mm-vet/mm-vet.json"
    images_dir = "images/mm-vet/images"
    output_path = "results/mmvet_results.json"
    
    # 运行benchmark
    run_mmvet_benchmark(mmvet_json_path, images_dir, output_path) 