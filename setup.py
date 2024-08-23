from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "").strip() for req in requirements]
        
        # Remove empty lines, comments, and '-e .'
        requirements = [req for req in requirements if req and not req.startswith('#') and req != '-e .']
    
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Akash",
    author_email="vishwakarmaakashav17@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)