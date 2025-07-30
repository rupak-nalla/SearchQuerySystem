import yaml

def load_yaml_fields(filepath: str):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

    extracted = []
    for entry in data:
        title = entry.get("Title", "N/A")
        description = entry.get("Natural Language Description", "N/A")
        hard = entry.get("Hard Criteria", [])
        soft = entry.get("Soft Criteria", [])

        extracted.append({
            "title": title,
            "description": description,
            "hard_criteria": hard,
            "soft_criteria": soft
        })

    return extracted


# def print_extracted_fields(filepath: str):
#     queries = load_yaml_fields(filepath)
#     for i, q in enumerate(queries):
#         print(f"\n--- Query {i + 1} ---")
#         print(f"Title: {q['title']}")
#         print(f"Description: {q['description']}")
#         print(f"Hard Criteria: {q['hard_criteria']}")
#         print(f"Soft Criteria: {q['soft_criteria']}")

