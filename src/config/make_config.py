import os
import re
from typing import Optional

import yaml
from dotenv import find_dotenv, load_dotenv
from loguru import logger


def _load_env() -> None:
    dotenv_path: Optional[str] = find_dotenv(filename='.env')
    if not dotenv_path:
        logger.warning(".env file not found using built-in methods. Falling back to default location")

        curr_dir: str = os.path.dirname(p=os.path.abspath(__file__))
        src_dir: str = os.path.dirname(p=curr_dir)
        root_dir: str = os.path.dirname(p=src_dir)
        dotenv_path: str = os.path.join(root_dir, ".env")
    
    if not os.path.exists(path=dotenv_path):
        logger.warning(f".env file not found at default location - {dotenv_path}")
        return
    
    logger.info(f".env file found at - {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

def make_config(config_file: str) -> dict:
    _load_env()
    
    config_kwargs = {}
    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    pattern = re.compile(r'\$\{([^}]+)\}')
    content = pattern.sub(lambda m: os.getenv(m.group(1), ''), content)
    
    config_kwargs = yaml.safe_load(stream=content)

    return config_kwargs


