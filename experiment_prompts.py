from src.llm_client import extract_features_stage1

queries = [
    "3 bed 2 bath colonial with 2 car garage",
    "Large lot, excellent kitchen, near downtown",
    "Fixer-upper with 1 bath and no garage"
]

print("Prompt Versioning Experiment")
print("="*60)
for q in queries:
    print(f"\nQuery: {q}")
    for ver in ["v1", "v2"]:
        result = extract_features_stage1(q, version=ver)
        print(f"  {ver}: {result}")
    print()
