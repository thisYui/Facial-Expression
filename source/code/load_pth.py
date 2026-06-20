import os
import sys

try:
    import torch
except Exception as e:
    print("ERROR: PyTorch is not installed or failed to import:", e)
    print("Install PyTorch (CPU) with: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(2)


def summarize(obj, max_items=20):
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"- dict with {len(keys)} keys. Showing up to {max_items} keys:")
        for k in keys[:max_items]:
            v = obj[k]
            if hasattr(v, "shape"):
                print(f"  {k}: tensor shape {v.shape}")
            else:
                print(f"  {k}: type {type(v)}")
    else:
        print(f"- object type: {type(obj)}")


def inspect_file(path):
    print('='*80)
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print("- File not found.")
        return
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print("- Failed to load with torch.load():", e)
        return

    print(f"- top-level type: {type(data)}")

    # Common patterns: state_dict (dict of parameter tensors), or a dict wrapper
    if isinstance(data, dict):
        # if it's a checkpoint with nested 'state_dict' key
        if 'state_dict' in data:
            print("- Contains key 'state_dict' -> showing its summary:")
            summarize(data['state_dict'])
        else:
            # Could be raw state_dict or something else
            summarize(data)
    else:
        # Could be a full model object (rare)
        print("- Non-dict object; attempting to inspect attributes...")
        attrs = [a for a in dir(data) if not a.startswith('__')]
        print(f"- Attributes count: {len(attrs)}. Showing first 20:")
        for a in attrs[:20]:
            try:
                val = getattr(data, a)
                print(f"  {a}: {type(val)}")
            except Exception:
                print(f"  {a}: <uninspectable>")


if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, 'result', 'ours_best.pth'),
        os.path.join(repo_root, 'result', 'ours_final.pth'),
    ]

    for p in candidates:
        inspect_file(p)

    print('='*80)
    print('Done.')
