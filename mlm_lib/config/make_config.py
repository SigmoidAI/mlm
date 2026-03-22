import os
import re
from typing import Any, Optional

import yaml
from dotenv import find_dotenv, load_dotenv
from loguru import logger


def _load_env() -> None:
    """Wrapper method for load_dotenv() method from dotenv library.

    Dynamically searches for .env file location using find_dotenv() method. If no path was found, it falls back to default location at root directory of the project.
    """
    dotenv_path: Optional[str] = find_dotenv(filename='.envfile')
    if not dotenv_path:
        logger.warning(".env file not found using built-in methods. Falling back to default location")

        curr_dir: str = os.path.dirname(p=os.path.abspath(__file__))
        src_dir: str = os.path.dirname(p=curr_dir)
        root_dir: str = os.path.dirname(p=src_dir)
        dotenv_path: str = os.path.join(root_dir, ".env")
    
    if not os.path.exists(path=dotenv_path):
        logger.warning(f".env file not found at default location - {dotenv_path}")
        return
    
    logger.success(f".env file found at - {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, override=True)

def make_config(config_file_path: str = None) -> dict:
    """Generates the config dict object from .yaml file.

    If no config file is provided, default configuration file is retrieved (cascade_models.yaml)

    Args:
        config_file_path (str, optional): Path to .yaml configuration file. Defaults to None.

    Returns:
        dict: Configuration dict object.
    """
    _load_env()
    
    if not config_file_path:
        config_file_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascade_models.yaml")
        logger.info(f"Configuration file not provided. Falling back to default location - {config_file_path}")
    
    config_kwargs: dict[str, Any] = {}
    with open(config_file_path, "r", encoding="utf-8") as f:
        content: str = f.read()
    
    pattern = re.compile(r'\$\{([^}]+)\}')
    content = pattern.sub(lambda m: os.getenv(m.group(1), ''), content)
    
    config_kwargs = yaml.safe_load(stream=content)

    logger.success("Configuration file loaded succesfully")
    return config_kwargs
