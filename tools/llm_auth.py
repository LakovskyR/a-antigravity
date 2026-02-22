"""
Multi-LLM Connectivity Test
Pings each configured LLM API to verify connectivity.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv

# Load environment variables from project root
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def test_claude_api(api_key: str) -> Tuple[bool, str]:
    """Test Claude API connectivity."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Say 'Hello from Claude'"}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return True, "Connected"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP Error: {e.response.status_code}"
    except Exception as e:
        return False, str(e)


def test_openai_api(api_key: str) -> Tuple[bool, str]:
    """Test OpenAI API connectivity."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say 'Hello from OpenAI'"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return True, "Connected"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP Error: {e.response.status_code}"
    except Exception as e:
        return False, str(e)


def test_kimi_api(api_key: str) -> Tuple[bool, str]:
    """Test Kimi API connectivity."""
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "moonshot-v1-8k",
        "messages": [{"role": "user", "content": "Say 'Hello from Kimi'"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return True, "Connected"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP Error: {e.response.status_code}"
    except Exception as e:
        return False, str(e)


def test_perplexity_api(api_key: str) -> Tuple[bool, str]:
    """Test Perplexity API connectivity."""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "sonar",
        "messages": [{"role": "user", "content": "Say 'Hello from Perplexity'"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return True, "Connected"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP Error: {e.response.status_code}"
    except Exception as e:
        return False, str(e)


def test_all_llms() -> Dict[str, Dict]:
    """
    Test all configured LLM providers.
    
    Returns:
        Dictionary with test results for each provider
    """
    print("=" * 60)
    print("MULTI-LLM CONNECTIVITY TEST")
    print("=" * 60)
    
    load_env()
    
    # Define providers and their test functions
    providers = {
        "claude": {
            "key_env": "CLAUDE_API_KEY",
            "test_func": test_claude_api,
            "description": "Anthropic Claude"
        },
        "openai": {
            "key_env": "OPENAI_API_KEY",
            "test_func": test_openai_api,
            "description": "OpenAI GPT"
        },
        "kimi": {
            "key_env": "KIMI_API_KEY",
            "test_func": test_kimi_api,
            "description": "Moonshot Kimi"
        },
        "perplexity": {
            "key_env": "PERPLEXITY_API_KEY",
            "test_func": test_perplexity_api,
            "description": "Perplexity AI"
        }
    }
    
    results = {}
    available = []
    unavailable = []
    not_configured = []
    
    print("\nTesting configured LLM providers:\n")
    
    for provider_id, config in providers.items():
        api_key = os.getenv(config["key_env"])
        placeholder = f"your_{provider_id}_api_key_here"
        
        if not api_key or api_key == placeholder:
            print(f"⚪ {provider_id.upper():12} — Not configured ({config['key_env']} missing)")
            results[provider_id] = {"status": "not_configured", "error": None}
            not_configured.append(provider_id)
            continue
        
        success, message = config["test_func"](api_key)
        
        if success:
            print(f"✅ {provider_id.upper():12} — Connected ({config['description']})")
            results[provider_id] = {"status": "available", "error": None}
            available.append(provider_id)
        else:
            print(f"❌ {provider_id.upper():12} — Failed: {message}")
            results[provider_id] = {"status": "failed", "error": message}
            unavailable.append(provider_id)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Available:     {len(available)} ({', '.join(available) if available else 'none'})")
    print(f"❌ Failed:        {len(unavailable)} ({', '.join(unavailable) if unavailable else 'none'})")
    print(f"⚪ Not configured: {len(not_configured)} ({', '.join(not_configured) if not_configured else 'none'})")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = test_all_llms()
    
    # Exit with error code if none are available
    any_available = any(r["status"] == "available" for r in results.values())
    sys.exit(0 if any_available else 1)
