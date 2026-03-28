import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def convert_oea():
    in_path = Path("data/openearth_agent/test.json")
    out_path = Path("data/openearthagent_eval_public.jsonl")
    
    if not in_path.exists():
        logging.warning(f"{in_path} not found.")
        return
        
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    converted = []
    for row in data:
        # Extract question
        q = ""
        for msg in row.get("conversation", []):
            if msg.get("from") == "human" and not msg.get("value", "").startswith("<AGENT_PROMPT>"):
                q = msg.get("value")
                break
        if not q and len(row.get("conversation", [])) > 2:
            q = row["conversation"][2].get("value", "")
            
        # Extract answer and tools
        ans = ""
        gt_tools = []
        for msg in row.get("conversation", []):
            if msg.get("from") == "gpt":
                try:
                    parsed = json.loads(msg.get("value", "{}"))
                    actions = parsed.get("actions", [])
                    if actions and actions[-1].get("name") == "Terminate":
                        ans = actions[-1].get("arguments", {}).get("ans", "")
                    
                    for act in actions:
                        if act.get("name") and act.get("name") != "Terminate":
                            gt_tools.append({"name": act.get("name"), "arguments": act.get("arguments", {})})
                except:
                    pass
                    
        converted.append({
            "id": row.get("idx"),
            "question": q,
            "answer": ans,
            "tools": gt_tools,
            "images": [f"data/openearth_agent/test/{img}" for img in row.get("images", [])]
        })
        
    with out_path.open("w", encoding="utf-8") as f:
        for c in converted:
            f.write(json.dumps(c) + "\n")
            
    logging.info(f"Converted {len(converted)} OEA samples to {out_path}")

def convert_thinkgeo():
    in_path = Path("data/thinkgeo/ThinkGeoBench.json")
    out_path = Path("data/thinkgeo_eval_public.jsonl")
    
    if not in_path.exists():
        logging.warning(f"{in_path} not found.")
        return
        
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    converted = []
    for k, row in data.items():
        q = row.get("dialogs", [{}])[0].get("content", "")
        
        gt = row.get("gt_answer")
        ans = ""
        if isinstance(gt, dict) and gt.get("whitelist"):
            ans = gt["whitelist"]
            
        # If no explicit gt, fallback to final assistant content
        if not ans:
            for msg in reversed(row.get("dialogs", [])):
                if msg.get("role") == "assistant" and msg.get("content"):
                    ans = msg["content"]
                    break

        gt_tools = []
        for msg in row.get("dialogs", []):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    if func.get("name"):
                        gt_tools.append({"name": func.get("name"), "arguments": func.get("arguments", {})})
        
        converted.append({
            "id": k,
            "question": q,
            "answer": ans,
            "tools": gt_tools,
            "images": [f"data/thinkgeo/{f.get('path')}" for f in row.get("files", []) if f.get("type") == "image"]
        })
        
    with out_path.open("w", encoding="utf-8") as f:
        for c in converted:
            f.write(json.dumps(c) + "\n")
            
    logging.info(f"Converted {len(converted)} ThinkGeo samples to {out_path}")

if __name__ == "__main__":
    convert_oea()
    convert_thinkgeo()
