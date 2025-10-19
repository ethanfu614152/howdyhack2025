from bs4 import BeautifulSoup
import requests
import json

base_url = "https://getinvolved.tamu.edu/organizations"
all_orgs = []  # master list for all data
page = 1  # start from page 1

while True:
    print(f"🔄 Fetching page {page}... ")
    url = f"{base_url}?page={page}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Collect all organization links on this page
    URL_list = [
        a["href"]
        for a in soup.find_all("a", href=True)
        if "/org/" in a["href"]
    ]

    # Remove duplicates
    URL_list = list(set(URL_list))

    if not URL_list:
        print("🚫 No more organizations found, stopping pagination. ")
        break  # exit the loop if no orgs found

    for org_url in URL_list:
        response = requests.get(org_url)
        soup = BeautifulSoup(response.text, "html.parser")

        org_slug = org_url.rstrip("/").split("/")[-1]

        org_data = {
            "url": org_url,
            "name": org_slug,
            "university": None,
            "description": None,
            "tags": [],
            "accepted_groups": []
        }

        # --- Name ---
        title_tag = soup.find("h1", class_="text-4xl font-semibold text-center leading-tight")
        if title_tag:
            org_data["name"] = title_tag.get_text(strip=True)

        # --- University affiliation ---
        university = soup.find("section", class_="mx-auto max-w-7xl px-6 lg:px-0 text-center relative")
        if university:
            works = university.find("div", class_="text-base2 italic text-gray-600 leading-8")
            if works:
                org_data["university"] = works.get_text(strip=True)

        # --- Description + tags ---
        section = soup.find("section", class_="mx-auto max-w-7xl px-6 lg:px-0 text-center relative")
        if section:
            paragraphs = section.find_all("p")
            if paragraphs:
                org_data["description"] = " ".join(p.get_text(strip=True) for p in paragraphs)
            else:
                text_div = section.find("div", class_="text-base2 pt-4")
                if text_div:
                    org_data["description"] = text_div.get_text(strip=True)

            tags = section.find_all("a")
            org_data["tags"] = [tag.get_text(strip=True) for tag in tags]

        # --- Accepted groups ---
        accepting = soup.find("section", class_="pt-14 lg:pt-16 mx-auto max-w-7xl contain")
        if accepting:
            p_tag = accepting.find("p")
            if p_tag:
                text = p_tag.get_text(strip=True)
                if "We accept:" in text:
                    accepted = text.replace("We accept:", "").strip()
                    org_data["accepted_groups"] = [item.strip() for item in accepted.split(",")]

        all_orgs.append(org_data)
        print(f"✅ Collected data for '{org_data['name']}' ")

    page += 1  # move to the next page

# --- Save all to ONE file ---
with open("tamu_orgs_combined.json", "w", encoding="utf-8") as f:
    json.dump(all_orgs, f, ensure_ascii=False, indent=4)

print(f"\n🎉 Saved all {len(all_orgs)} orgs to tamu_orgs_combined.json ")
