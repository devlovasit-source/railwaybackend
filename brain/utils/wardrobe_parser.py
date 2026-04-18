# utils/wardrobe_parser.py
import re

def extract_and_clean_response(llama_text: str, wardrobe: list) -> dict:
    """
    Parses CHIPS, PACK_LIST, and STYLE_BOARD from the LLM response.
    Relies on style_engine.py for strict outfit validation.
    """
    response_data = {
        "cleaned_text": llama_text,
        "chips": [],
        "pack_tag": "",
        "board_tag": ""
    }

    # 1. Parse Chips
    chip_match = re.search(r'\[CHIPS?:(.*?)\]', response_data["cleaned_text"], re.IGNORECASE)
    if chip_match:
        response_data["chips"] = [c.strip() for c in chip_match.group(1).split(',') if c.strip()]
    response_data["cleaned_text"] = re.sub(r'\[CHIPS?:.*?\]', '', response_data["cleaned_text"], flags=re.IGNORECASE).strip()

    # 2. Extract Packing List
    pack_match = re.search(r'\[?PACK_LIST:\s*(.*?)(?:\]|\n|$)', response_data["cleaned_text"], re.IGNORECASE)
    if pack_match:
        # Just extract exactly what the backend provided
        raw_pack_str = pack_match.group(1).strip()
        response_data["pack_tag"] = f"[PACK_LIST: {raw_pack_str}]"
        response_data["cleaned_text"] = re.sub(r'\[?PACK_LIST:.*?(\]|\n|$)', '', response_data["cleaned_text"], flags=re.IGNORECASE).strip()

    # 3. Extract Style Board 
    board_match = re.search(r'\[?STYLE_BOARD:\s*(.*?)(?:\]|\n|$)', response_data["cleaned_text"], re.IGNORECASE)
    if board_match:
        # The Style Engine already guarantees this string contains perfectly validated IDs.
        # We just need to extract the string and strip it from the text.
        raw_items_str = board_match.group(1).strip()
        
        if raw_items_str:
            response_data["board_tag"] = f"[STYLE_BOARD: {raw_items_str}]"
            
        response_data["cleaned_text"] = re.sub(r'\[?STYLE_BOARD:.*?(\]|\n|$)', '', response_data["cleaned_text"], flags=re.IGNORECASE).strip()

    # 4. Final Text Cleanup (Removing stray IDs and artifacts)
    for item in wardrobe:
        item_id = str(item.get("$id") or item.get("id", ""))
        if item_id and item_id in response_data["cleaned_text"]:
            response_data["cleaned_text"] = response_data["cleaned_text"].replace(item_id, "")
    
    response_data["cleaned_text"] = re.sub(r'\b(item|items|id1|id2|id3|id4)\b', '', response_data["cleaned_text"], flags=re.IGNORECASE)
    response_data["cleaned_text"] = re.sub(r'\(\s*[,\s]*\)', '', response_data["cleaned_text"]) 
    response_data["cleaned_text"] = re.sub(r'\s{2,}', ' ', response_data["cleaned_text"]).strip() 

    return response_data