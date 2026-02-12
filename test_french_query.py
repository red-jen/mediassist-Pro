"""Test RAG system with French queries to match the French PDF content."""
import requests
import json

# First login to get token
login_data = {
    "username": "admin",
    "password": "admin123"
}

response = requests.post("http://localhost:8000/api/v1/auth/login", json=login_data)
if response.status_code == 200:
    token = response.json()["access_token"]
    print("✅ Login successful")
else:
    print(f"❌ Login failed: {response.text}")
    exit(1)

# Test with French query matching the WHO laboratory maintenance manual
headers = {"Authorization": f"Bearer {token}"}

# French queries that should match the WHO laboratory maintenance manual
french_queries = [
    "Comment faire la maintenance des appareils de laboratoire?",
    "Quels sont les protocoles d'entretien des équipements?", 
    "Procédures de calibrage des instruments de laboratoire",
    "Maintenance préventive des centrifugeuses",
    "Entretien des appareils de mesure"
]

print("\n=== Testing French Queries ===")
for query in french_queries:
    print(f"\nTesting: '{query}'")
    
    query_data = {"question": query}
    response = requests.post("http://localhost:8000/api/v1/query/ask", 
                           json=query_data, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        if "sources" in result and result["sources"]:
            print(f"✅ Found {len(result['sources'])} relevant documents")
            print(f"Answer: {result['answer'][:200]}...")
        else:
            print("❌ No relevant documents found")
    else:
        print(f"❌ Query failed: {response.text}")