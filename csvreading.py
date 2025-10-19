import json
import csv

# Load JSON file
with open("tamu_orgs_combined.json", "r", encoding="utf-8") as f:
    orgs = json.load(f)

csv_file = "tamu_orgs_clean.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["url", "name", "description", "tags", "accepted_groups"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for org in orgs:
        # --- Ensure name exists ---
        name = org.get("name")
        description = org.get("description", "")
        url_slug = org.get("url", "").rstrip("/").split("/")[-1]

        # Flatten lists for CSV
        tags = ", ".join(org.get("tags", []))
        accepted_groups = ", ".join(org.get("accepted_groups", []))

        writer.writerow({
            "url": org.get("url", ""),
            "name": name,
            "description": description,
            "tags": tags,
            "accepted_groups": accepted_groups
        })

print(f"🎉 CSV file saved as {csv_file} with {len(orgs)} organizations 😭🙏😭🙏😭🙏😭🙏😭🙏")
