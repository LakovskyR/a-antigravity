"""
GitHub Authentication Test
Verifies connectivity to GitHub API and repository access.
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


def test_github_connection() -> bool:
    """
    Test GitHub API connectivity and repository access.
    
    Returns:
        True if connection successful, False otherwise
    """
    print("=" * 60)
    print("GITHUB CONNECTIVITY TEST")
    print("=" * 60)
    
    load_env()
    
    token = os.getenv("GITHUB_TOKEN")
    repo_url = os.getenv("GITHUB_REPO")
    
    # Check credentials
    if not token or token == "your_github_token_here":
        print("\n❌ GITHUB_TOKEN not configured")
        print("   Please set GITHUB_TOKEN in your .env file")
        return False
    
    if not repo_url or repo_url == "your_repo_url_here":
        print("\n⚠️  GITHUB_REPO not configured")
        print("   Will test API access only (no repo verification)")
        test_repo = False
    else:
        test_repo = True
        print(f"\n1. Testing GitHub API access...")
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Test 1: API authentication
    try:
        response = requests.get("https://api.github.com/user", headers=headers, timeout=30)
        response.raise_for_status()
        user_data = response.json()
        print(f"   ✅ Authenticated as: {user_data.get('login')}")
    except requests.exceptions.HTTPError as e:
        print(f"   ❌ Authentication failed: HTTP {e.response.status_code}")
        if e.response.status_code == 401:
            print("   Token is invalid or expired")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Repository access (if configured)
    if test_repo:
        print(f"\n2. Testing repository access...")
        print(f"   Repository: {repo_url}")
        
        # Parse repo URL to get owner/repo format
        # Handle various URL formats
        repo_path = repo_url
        for prefix in ["https://github.com/", "git@github.com:", "github.com/"]:
            if repo_path.startswith(prefix):
                repo_path = repo_path[len(prefix):]
        
        # Remove .git suffix if present
        if repo_path.endswith(".git"):
            repo_path = repo_path[:-4]
        
        try:
            repo_api_url = f"https://api.github.com/repos/{repo_path}"
            response = requests.get(repo_api_url, headers=headers, timeout=30)
            response.raise_for_status()
            repo_data = response.json()
            print(f"   ✅ Repository access confirmed")
            print(f"   Name: {repo_data.get('full_name')}")
            print(f"   Default branch: {repo_data.get('default_branch')}")
            
            # Check permissions
            permissions = repo_data.get("permissions", {})
            if permissions.get("push"):
                print(f"   ✅ Write access confirmed")
            else:
                print(f"   ⚠️  Read-only access (no push permission)")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"   ❌ Repository not found or no access")
            else:
                print(f"   ❌ Error: HTTP {e.response.status_code}")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("GITHUB: ✅ CONNECTED")
    print("=" * 60)
    return True


def get_repo_info() -> dict:
    """
    Get current repository information if available.
    
    Returns:
        Dictionary with repo details or empty dict if not configured
    """
    load_env()
    
    token = os.getenv("GITHUB_TOKEN")
    repo_url = os.getenv("GITHUB_REPO")
    
    if not token or not repo_url:
        return {}
    
    # Parse repo URL
    repo_path = repo_url
    for prefix in ["https://github.com/", "git@github.com:", "github.com/"]:
        if repo_path.startswith(prefix):
            repo_path = repo_path[len(prefix):]
    
    if repo_path.endswith(".git"):
        repo_path = repo_path[:-4]
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(
            f"https://api.github.com/repos/{repo_path}",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except:
        return {}


if __name__ == "__main__":
    success = test_github_connection()
    sys.exit(0 if success else 1)
