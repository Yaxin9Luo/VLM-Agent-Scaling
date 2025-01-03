from setuptools import setup, find_packages

setup(
    name="vlm_agent_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv==1.0.0",
        "requests==2.31.0",
        "pydantic==2.5.2",
        "loguru==0.7.2",
        "pillow==10.1.0",
    ],
) 