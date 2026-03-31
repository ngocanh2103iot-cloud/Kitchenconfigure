import json
import sys

nb_path = r"c:\Users\Administrator\Desktop\Train\Kitchenconfigure\kitchen\Bản_sao_của_kitchen2.ipynb"
try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
except Exception as e:
        pass
with open("nb_extract.py", "w", encoding="utf-8") as out:
    try:
        out.write("Code cells:\n" + "=" * 40 + "\n")
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                if "mfcc" in source.lower() or "librosa" in source.lower() or "extract" in source.lower():
                    out.write(source)
                    out.write("\n# --- END OF CELL ---\n")
    except Exception as e:
        out.write(f"Error: {e}\n")
