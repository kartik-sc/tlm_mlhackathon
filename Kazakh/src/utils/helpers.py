import json

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)