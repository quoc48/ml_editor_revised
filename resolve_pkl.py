import dill

file_path = "models/vectorizer_1.pkl"

with open(file_path, "rb") as f:
    content = f.read()

# Remove BOM if present
if content[:3] == b'\xef\xbf\xbd':
    content = content[3:]

try:
    import pickle
    data = pickle.loads(content)
    print("File loaded successfully")
except Exception as e:
    print(f"Failed to load file: {e}")

