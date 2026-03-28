import json
import re

try:
    with open('Bản_sao_của_kitchen2.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    found_scaler = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'scaler' in source.lower() or 'scale' in source.lower() or 'mean' in source.lower():
                print(f"--- Code Cell ---")
                print(source)
                found_scaler = True
                
    if not found_scaler:
        print("Không tìm thấy lệnh nào chứa từ 'scaler' trong notebook.")
except Exception as e:
    print("Error:", e)
