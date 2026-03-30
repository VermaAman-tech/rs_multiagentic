from setuptools import find_packages, setup

setup(
    name="magrf",
    version="0.1.0",
    description="Multi-Agent Geospatial Reasoning Framework",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi==0.115.12",
        "uvicorn==0.34.0",
        "pydantic==2.10.6",
        "httpx==0.28.1",
        "pyyaml==6.0.2",
        "requests==2.32.3",
        "Pillow==11.1.0",
        "numpy==1.26.4",
        "torch==2.6.0",
    ],
)
