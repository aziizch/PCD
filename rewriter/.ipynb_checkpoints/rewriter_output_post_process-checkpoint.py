"""
Post-processing: Extract Key Information from Rewriter's Output
Compatible with [start_of_analysis] ... [end_of_analysis] format
"""

import json
import re
import pandas as pd

# --------------------------------------------------
# 1. Extract analysis block
# --------------------------------------------------

def extract_analysis_block(text):
    if not isinstance(text, str):
        return ""
    pattern = r'\[start_of_analysis\]\s*(.*?)\s*\[end_of_analysis\]'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# --------------------------------------------------
# 2. Extract code entities
# --------------------------------------------------

def extract_code_entities(text):
    if not isinstance(text, str):
        return []
    entities = re.findall(r'`([^`]+)`', text)
    return list(set(entities))

# --------------------------------------------------
# 3. Extract keywords
# --------------------------------------------------

def extract_keywords(text):
    if not isinstance(text, str):
        return []
    words = re.findall(r'\b[A-Za-z_]{4,}\b', text)
    blacklist = {
        "this", "that", "with", "from", "need", "using", "include",
        "where", "when", "such", "will", "have", "should", "must",
        "also", "more", "than", "they", "them", "their"
    }
    return list(set(w for w in words if w.lower() not in blacklist))

# --------------------------------------------------
# 4. Main
# --------------------------------------------------

if __name__ == "__main__":

    INPUT_FILE = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\rewriter\test_rewriter_output.json"
    OUTPUT_FILE = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\rewriter\rewriter_output.json"
    PROCESSED_FILE = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\rewriter\test_rewriter_output_processed.json"

    # Load data
    test_basic_info = pd.read_json(INPUT_FILE)

    # استخراج التحليل
    test_basic_info["rewriter_inferer_output"] = test_basic_info["rewriter_extractor"].apply(
        extract_analysis_block
    )

    # استخراج الكيانات والكلمات المفتاحية
    test_basic_info["rewriter_extractor_output_entity"] = test_basic_info["rewriter_inferer_output"].apply(
        extract_code_entities
    )
    test_basic_info["rewriter_extractor_output_keyword"] = test_basic_info["rewriter_inferer_output"].apply(
        extract_keywords
    )

    # Build final dict
    rewriter_output_dict = {}
    error_case = []

    for idx, item in test_basic_info.iterrows():
        instance_id = item.get("instance_id", str(idx))
        entities = item.rewriter_extractor_output_entity
        keywords = item.rewriter_extractor_output_keyword
        query = item.rewriter_inferer_output
        if entities or keywords or query:
            rewriter_output_dict[instance_id] = {
                "code_entity": entities,
                "keyword": keywords,
                "query": query,
            }
        else:
            error_case.append(instance_id)

    # Save final JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(rewriter_output_dict, f, indent=2, ensure_ascii=False)

    # Save processed dataset
    test_basic_info.to_json(PROCESSED_FILE, index=False)

    # Debug info
    print("Extraction terminee")
    print(f"Items valides : {len(rewriter_output_dict)}")
    print(f"Items vides   : {len(error_case)}")








