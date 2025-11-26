import json
from pathlib import Path

def generate_object_weights():
    # Use the script's location to find the root directory
    script_dir = Path(__file__).resolve().parent
    root = script_dir / "datasets_nonlabel"
    target_categories = ["Cookies", "Crab", "Danta", "红薯", "面点"]
    
    weights = {
        "default_weight": 20.0,
        "classes": {}
    }
    
    if not root.exists():
        print(f"Error: {root} does not exist.")
        return

    for category in target_categories:
        cat_path = root / category
        if not cat_path.exists():
            continue
            
        # Iterate over subdirectories (subclasses)
        for sub_path in cat_path.iterdir():
            if sub_path.is_dir():
                # Construct key: category/subclass
                # Normalize to match sam_seg.py logic: lowercase, forward slashes
                key = f"{category}/{sub_path.name}".lower().replace("\\", "/")
                weights["classes"][key] = 20.0
                print(f"Added: {key}")

    output_file = Path("weight_per_object.json")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully generated {output_file}")

if __name__ == "__main__":
    generate_object_weights()
