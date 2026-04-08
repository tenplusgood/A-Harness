"""
API Client for OpenAI-compatible chat/completions API.

Supports loading configuration from config.py, environment variables,
and explicit parameter overrides.
"""

import importlib.util
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _load_dotenv_if_exists() -> None:
    """Load .env from project root if available."""
    if load_dotenv is None:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load_dotenv_if_exists()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
     config.py 
    
    Args:
        config_path: ， None， config.py
    
    Returns:
        ， API_KEY, API_BASE_URL, DEFAULT_MODEL 
    """
    if config_path is None:
        possible_paths = [
            Path(__file__).resolve().parent.parent / "config.py",  # A-Harness/config.py
            Path.cwd() / "config.py",
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None or not config_path.exists():
            return {}
    
    if not config_path.exists():
        return {}
    
    try:
        spec = importlib.util.spec_from_file_location("aff_config", config_path)
        if spec and spec.loader:
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)
            
            config = {}
            for key in [
                "API_KEY", "API_BASE_URL", "DEFAULT_MODEL", "DEFAULT_OUTPUT_DIR",
                "QWEN35_API_BASE_URL", "QWEN35_API_KEY", "QWEN35_MODEL_NAME",
            ]:
                if hasattr(cfg, key):
                    config[key] = getattr(cfg, key)
            
            return config
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return {}
    
    return {}


def get_default_api_config() -> Tuple[Optional[str], Optional[str]]:
    """
     API （ config.py ）
    
    Returns:
        (api_key, api_base_url) 
    """
    #  config.py 
    config = load_config()
    
    api_key = config.get("API_KEY") or os.getenv("API_KEY")
    api_base_url = config.get("API_BASE_URL") or os.getenv("API_BASE_URL")
    
    return api_key, api_base_url


def get_qwen35_api_config() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
     Qwen3.5 PAI-EAS  API 

    Returns:
        (api_key, api_base_url, model_name) 
    """
    config = load_config()

    api_key = config.get("QWEN35_API_KEY") or os.getenv("QWEN35_API_KEY")
    api_base_url = config.get("QWEN35_API_BASE_URL") or os.getenv("QWEN35_API_BASE_URL")
    model_name = config.get("QWEN35_MODEL_NAME") or os.getenv("QWEN35_MODEL_NAME")

    return api_key, api_base_url, model_name


def is_qwen35_model(model_name: str) -> bool:
    """ Qwen3.5（ PAI-EAS ）"""
    if not model_name:
        return False
    name = model_name.lower().replace("-", "").replace("_", "").replace(".", "")
    return "qwen35" in name


def normalize_api_url(base_url: str) -> str:
    """
     API URL，
    
    Args:
        base_url:  base URL
    
    Returns:
         URL
    """
    url = base_url.rstrip("/")
    
    #  URL  /v1， OpenAI  API，
    if "/v1" not in url and not url.endswith("/chat") and not url.endswith("/completions"):
        known_domains = [
            "api.openai.com", "api.vveai.com", "api.anthropic.com",
            "pai-eas.aliyuncs.com",
        ]
        if any(domain in url for domain in known_domains):
            if not url.endswith("/v1"):
                url = f"{url}/v1"
    
    return url


class APIClient:
    """
    API Client for OpenAI-compatible chat/completions API
    
    ：
    -  config.py 
    - 
    - 
    - 
    - 
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config_path: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize API client
        
        Args:
            api_key: API key for authentication ( config.py )
            base_url: Base URL for the API ( config.py )
            config_path: （）
            max_retries: （ 3）
            retry_delay: （， 1.0）
        """
        # ： >  > config.py
        if api_key is None:
            api_key, _ = get_default_api_config()
        
        if base_url is None:
            _, base_url = get_default_api_config()
        
        # ， config_path 
        if (api_key is None or base_url is None) and config_path:
            config = load_config(config_path)
            if api_key is None:
                api_key = config.get("API_KEY")
            if base_url is None:
                base_url = config.get("API_BASE_URL")
        
        self.api_key = api_key
        self.base_url = normalize_api_url(base_url) if base_url else None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.base_url:
            raise ValueError(
                "base_url is required. Please provide it via:\n"
                "  1. Parameter: APIClient(base_url='...')\n"
                "  2. Environment variable: export API_BASE_URL='...'\n"
                "  3. config.py: API_BASE_URL = '...'"
            )
        
        if not self.api_key:
            print("Warning: No API key provided. Some APIs may require authentication.")
    
    def call(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST",
        retry_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Make an API call with automatic retry
        
        Args:
            endpoint: API endpoint (e.g., "chat/completions")
            data: Request data
            method: HTTP method (default: "POST")
            retry_on_error: （ True）
        
        Returns:
            Response JSON as dictionary
        
        Raises:
            requests.HTTPError: If the API call fails after all retries
            requests.RequestException: If there's a network error
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        last_exception = None
        
        for attempt in range(self.max_retries if retry_on_error else 1):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    timeout=300,  # 5 minutes timeout for long-running requests
                )
        
                response.raise_for_status()
                
                return response.json()

            except requests.HTTPError as e:
                last_exception = e
                
                # （ 400 Bad Request），
                if response.status_code in [400, 401, 403, 404]:
                    break
                
                #  429 (Rate Limited) / 5xx ，
                if attempt < self.max_retries - 1:
                    # 429 ，
                    if response.status_code == 429:
                        base_wait = self.retry_delay * (3 ** attempt)  # 
                        wait_time = min(base_wait + random.uniform(1.0, 3.0), 60)
                    else:
                        base_wait = self.retry_delay * (2 ** attempt)
                        wait_time = base_wait + random.uniform(0.3, 1.0)  #  jitter 
                    print(f"API call failed: HTTP {response.status_code} "
                          f"(attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
            except requests.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt) + random.uniform(0.3, 1.0)
                    print(f"Network error (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
        
        error_msg = f"API call failed after {self.max_retries} attempts"
        
        if isinstance(last_exception, requests.HTTPError):
            error_msg += f": HTTP {last_exception.response.status_code} for {url}"
            try:
                error_body = last_exception.response.json()
                error_msg += f"\nResponse body: {json.dumps(error_body, indent=2)}"
            except:
                error_text = last_exception.response.text[:500]
                error_msg += f"\nResponse text: {error_text}"
        else:
            error_msg += f": {str(last_exception)}"
        
        raise requests.HTTPError(error_msg) from last_exception
    
    def chat_completions(
        self,
        messages: list,
        model: str,
        tools: Optional[list] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        ： chat/completions endpoint
        
        Args:
            messages: 
            model: 
            tools: （）
            tool_choice: （ "auto"）
            **kwargs: （ temperature, max_tokens ）
        
        Returns:
            API 
        """
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        if tools:
            data["tools"] = tools
            data["tool_choice"] = tool_choice
        
        return self.call("chat/completions", data)
