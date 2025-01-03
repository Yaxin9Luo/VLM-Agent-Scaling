# VLM Agent Pipeline

这是一个基于视觉语言模型(VLM)的多Agent处理pipeline系统。系统包含三层结构：
1. 第一层：专用Agents
   - OCR Agent: 处理图像文字识别任务
   - Relation Agent: 处理图像关系分析任务
   - Common Agent: 处理通用图像理解任务

2. 第二层：Refiner Agents
   - 三个Refiner Agent分别优化和改进第一层agents的输出

3. 第三层：Retriever Agent
   - 整合所有Refiner Agents的输出，生成最终答案

## 项目结构
```
vlm_agent_pipeline/
├── agents/                 # Agent实现
│   ├── base/              # 基础Agent类
│   ├── level1/            # 第一层专用Agents
│   ├── level2/            # 第二层Refiner Agents
│   └── level3/            # 第三层Retriever Agent
├── config/                # 配置文件
├── core/                  # 核心功能实现
├── utils/                 # 工具函数
└── examples/              # 示例代码
```

## 安装
```bash
pip install -r requirements.txt
```

## 使用方法
1. 配置环境变量：
```bash
cp .env.example .env
# 编辑.env文件，添加必要的API密钥
```

2. 运行示例：
```bash
python examples/ocr_example.py
``` 