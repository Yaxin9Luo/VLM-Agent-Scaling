import os
import json
from tqdm import tqdm
from loguru import logger
from run_pipeline import run_pipeline
from typing import Dict, List, Any
from pathlib import Path

def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison"""
    return answer.lower().strip().replace(" ", "")

def check_answer(pred: str, gt: str) -> bool:
    """Check if prediction matches any of the ground truth answers"""
    pred = normalize_answer(pred)
    
    # Handle OR/AND cases in ground truth
    if "<OR>" in gt:
        gt_options = gt.split("<OR>")
        return any(pred == normalize_answer(opt) for opt in gt_options)
    elif "<AND>" in gt:
        gt_parts = gt.split("<AND>")
        return all(normalize_answer(part) in pred for part in gt_parts)
    else:
        return pred == normalize_answer(gt)

def load_existing_results(output_file: str) -> Dict:
    """Load existing results if any"""
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def run_mmvet_v2_test(dataset_path: str, image_dir: str):
    """Run evaluation on MM-VET-V2 dataset"""
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 确保输出目录存在
    output_file = "results/mmvet_v2_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 加载已有结果
    results = load_existing_results(output_file)
    correct_count = 0
    total_count = 0
    
    # Process each question
    for qid, data in tqdm(dataset.items(), desc="Processing MM-VET-V2"):
        # Skip if already processed
        if qid in results:
            logger.info(f"Skipping {qid} (already processed)")
            continue
            
        question = data["question"]
        ground_truth = data["answer"]
        capabilities = data["capability"]
        
        # Extract image filename from question
        img_filename = question.split("<IMG>")[1]
        image_path = os.path.join(image_dir, img_filename)
        
        # Clean question
        question = question.split("<IMG>")[0].strip()
        
        try:
            # Run pipeline
            logger.info(f"\nProcessing {qid}: {question}")
            output = run_pipeline(image_path, question)
            
            # Store result in the same format as mmvet_test
            results[qid] = output.result
            
            # Save results after each question
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            # Check answer for statistics
            is_correct = check_answer(output.result, ground_truth)
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Log progress
            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
            logger.info(f"Current Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
            logger.info(f"Question: {question}")
            logger.info(f"Answer: {output.result}")
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {qid}: {str(e)}")
            # Save empty result if processing failed
            results[qid] = ""
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            continue
    
    # Calculate final metrics
    final_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    # Calculate accuracy by capability
    capability_metrics = {}
    for cap in ["ocr", "math", "spat", "rec"]:
        cap_questions = [(qid, data) for qid, data in dataset.items() if cap in data["capability"]]
        if cap_questions:
            cap_correct = sum(1 for qid, data in cap_questions if qid in results and check_answer(results[qid], data["answer"]))
            cap_accuracy = (cap_correct / len(cap_questions) * 100)
            capability_metrics[cap] = {
                "accuracy": cap_accuracy,
                "count": len(cap_questions)
            }
    
    # Log final results
    logger.info("\n=== Final Results ===")
    logger.info(f"Overall Accuracy: {final_accuracy:.2f}% ({correct_count}/{total_count})")
    logger.info("\nAccuracy by Capability:")
    for cap, metrics in capability_metrics.items():
        logger.info(f"{cap}: {metrics['accuracy']:.2f}% ({metrics['count']} questions)")
    
    return results

if __name__ == "__main__":
    dataset_path = "/root/autodl-tmp/VLM-Agent-Scaling/images/mm-vet-v2/mm-vet-v2.json"
    image_dir = "/root/autodl-tmp/VLM-Agent-Scaling/images/mm-vet-v2/images"
    run_mmvet_v2_test(dataset_path, image_dir) 