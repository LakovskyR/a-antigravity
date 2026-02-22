"""
SharePoint Authentication Test
Verifies connectivity to Microsoft SharePoint API.
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


def get_sharepoint_access_token(client_id: str, client_secret: str, tenant_id: str) -> str:
    """
    Obtain Microsoft Graph API access token using client credentials flow.
    
    Args:
        client_id: SharePoint application client ID
        client_secret: SharePoint application client secret
        tenant_id: Microsoft tenant ID
        
    Returns:
        Access token string
        
    Raises:
        requests.RequestException: If authentication fails
    """
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default"
    }
    
    response = requests.post(url, data=data)
    response.raise_for_status()
    
    return response.json()["access_token"]


def list_sharepoint_files(access_token: str, site_id: str, drive_path: str = "/") -> list:
    """
    List files in a SharePoint drive.
    
    Args:
        access_token: Valid Microsoft Graph access token
        site_id: SharePoint site ID (format: {hostname},{site-id},{web-id})
        drive_path: Path to list files from
        
    Returns:
        List of file dictionaries with name and webUrl
    """
    # Get default drive for site
    drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    response = requests.get(drives_url, headers=headers)
    response.raise_for_status()
    
    drives = response.json().get("value", [])
    if not drives:
        return []
    
    # Use first drive (typically Documents)
    drive_id = drives[0]["id"]
    
    # List files in drive
    files_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:{drive_path}:/children"
    response = requests.get(files_url, headers=headers)
    response.raise_for_status()
    
    return response.json().get("value", [])


def test_sharepoint_connection():
    """
    Test SharePoint connectivity and list files in test directory.
    """
    print("=" * 60)
    print("SHAREPOINT CONNECTIVITY TEST")
    print("=" * 60)
    
    load_env()
    
    # Check for required credentials
    client_id = os.getenv("SHAREPOINT_CLIENT_ID")
    client_secret = os.getenv("SHAREPOINT_SECRET")
    tenant_id = os.getenv("SHAREPOINT_TENANT_ID")
    
    missing = []
    if not client_id or client_id == "your_sharepoint_client_id_here":
        missing.append("SHAREPOINT_CLIENT_ID")
    if not client_secret or client_secret == "your_sharepoint_secret_here":
        missing.append("SHAREPOINT_SECRET")
    if not tenant_id or tenant_id == "your_tenant_id_here":
        missing.append("SHAREPOINT_TENANT_ID")
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("   Please set these in your .env file.")
        return False
    
    try:
        # Test authentication
        print("\n1. Testing authentication...")
        token = get_sharepoint_access_token(client_id, client_secret, tenant_id)
        print(f"   ✅ Authentication successful")
        print(f"   Token received: {token[:20]}...{token[-10:]}")
        
        # Note: Site ID needed for file listing
        print("\n2. File listing test...")
        print("   ⚠️  To list files, provide a SharePoint site ID")
        print("   Format: {hostname},{site-id},{web-id}")
        print("   Skipping file list test (no site ID configured)")
        
        print("\n" + "=" * 60)
        print("SHAREPOINT: ✅ CONNECTED (authentication only)")
        print("=" * 60)
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Authentication failed: {e}")
        if e.response is not None:
            print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_sharepoint_connection()
    sys.exit(0 if success else 1)
