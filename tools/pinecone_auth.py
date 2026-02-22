"""
Pinecone Authentication Test
Verifies connectivity to Pinecone vector database.
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from project root
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def test_pinecone_connection() -> bool:
    """
    Test Pinecone API connectivity and list available indexes.
    
    Returns:
        True if connection successful, False otherwise
    """
    print("=" * 60)
    print("PINECONE CONNECTIVITY TEST")
    print("=" * 60)
    
    load_env()
    
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    
    # Check credentials
    missing = []
    if not api_key or api_key == "your_pinecone_api_key_here":
        missing.append("PINECONE_API_KEY")
    if not environment or environment == "your_pinecone_environment_here":
        missing.append("PINECONE_ENVIRONMENT")
    
    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("   Please set these in your .env file.")
        return False
    
    print(f"\n1. Testing Pinecone API...")
    print(f"   Environment: {environment}")
    
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Test 1: List indexes
    try:
        # Pinecone control plane API
        response = requests.get(
            "https://api.pinecone.io/indexes",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        indexes = response.json().get("indexes", [])
        print(f"   ✅ API connection successful")
        print(f"   Available indexes: {len(indexes)}")
        
        if indexes:
            print("\n   Indexes:")
            for idx in indexes:
                name = idx.get("name", "unknown")
                dimension = idx.get("dimension", "?")
                metric = idx.get("metric", "?")
                print(f"   - {name} (dim: {dimension}, metric: {metric})")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print(f"   ❌ Authentication failed: Invalid API key")
        elif e.response.status_code == 403:
            print(f"   ❌ Forbidden: Check API key permissions")
        else:
            print(f"   ❌ HTTP Error: {e.response.status_code}")
            try:
                error_detail = e.response.json()
                print(f"      Details: {error_detail}")
            except:
                pass
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Serverless/Regional check
    print(f"\n2. Checking Pinecone environment...")
    try:
        # Try to describe a specific index if any exist
        if indexes:
            first_index = indexes[0]["name"]
            response = requests.get(
                f"https://api.pinecone.io/indexes/{first_index}",
                headers=headers,
                timeout=30
            )
            if response.status_code == 200:
                index_info = response.json()
                status = index_info.get("status", {}).get("state", "unknown")
                print(f"   Index '{first_index}' status: {status}")
                print(f"   ✅ Index access confirmed")
    except Exception as e:
        print(f"   ⚠️  Could not fetch index details: {e}")
    
    print("\n" + "=" * 60)
    print("PINECONE: ✅ CONNECTED")
    print("=" * 60)
    return True


def list_pinecone_indexes() -> list:
    """
    Get list of available Pinecone indexes.
    
    Returns:
        List of index dictionaries
    """
    load_env()
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or api_key == "your_pinecone_api_key_here":
        return []
    
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://api.pinecone.io/indexes",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("indexes", [])
    except:
        return []


if __name__ == "__main__":
    success = test_pinecone_connection()
    sys.exit(0 if success else 1)
