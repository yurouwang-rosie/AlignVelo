from setuptools import setup, find_packages

setup(
    name="AlignVelo",
    version="0.1.1",
    author="YurouWang",
    author_email="rapunzel@sjtu.edu.cn",
    description="A deep learning framework for batch-consistent RNA velocity estimation.",
    long_description=open("README.md").read(), 
    long_description_content_type="text/markdown",
    url="https://github.com/yurouwang-rosie/AlignVelo",
    packages=find_packages(), 
    install_requires=[
        "numpy==1.21.1", 
 	    "numba==0.53.1",
	    "pandas<2.0.0",
        "scvelo>=0.2.4,<0.3",
	    'torch>=1.9.1',
        'scanpy>=1.7.1',
        'anndata>=0.7.5',
        'scipy>=1.5.2',
        'tqdm>=4.32.2',
        'scikit-learn>=0.24.1',
	    'matplotlib>=3.3,<3.6',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8,<3.10",
    entry_points={
        "console_scripts": [
            "AlignVelo=AlignVelo.main:main",  # 创建命令行工具
        ],
    },
)