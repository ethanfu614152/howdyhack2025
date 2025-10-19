import requests
from bs4 import BeautifulSoup

url = "https://getinvolved.tamu.edu/organizations"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Collect all organization links
org_links = [
    a["href"]
    for a in soup.find_all("a", href=True)
    if "/org/" in a["href"]
]

# Remove duplicates
org_links = list(set(org_links))

for org in org_links:
    print(org)