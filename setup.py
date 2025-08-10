from setuptools import setup
from pathlib import Path

def load_requirements():
    # Path to requirements.txt inside the package
    req_path = Path(__file__).parent / "onboarding_agent" / "requirements.txt"
    with open(req_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f 
               if line.strip() and not line.startswith('#')]

setup(
    name="onboarding_agent",
    version="0.1.0",
    packages=[
        "onboarding_agent",
        "onboarding_agent.api",
        "onboarding_agent.api.v1",
        "onboarding_agent.service",
        "onboarding_agent.core",
        "onboarding_agent.data",
        "onboarding_agent.db",
    ],
    install_requires=load_requirements(),
    python_requires='>=3.8',
    include_package_data=True, 
)