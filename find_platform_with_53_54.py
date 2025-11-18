import json

platforms_to_check = [
    ('Platform/2_Platform.json', 'Platform 2'),
    ('Platform/5_Platform.json', 'Platform 5'),
    ('Platform/3_Platform.json', 'Platform 3'),
    ('Platform/EdgeAI-Trust_Platform.json', 'EdgeAI-Trust'),
]

print("Looking for processors 53 and 54...\n")

for platform_path, name in platforms_to_check:
    try:
        with open(platform_path) as f:
            plat = json.load(f)
            proc_ids = sorted([n['id'] for n in plat['platform']['nodes'] if not n.get('is_router', False)])
            has_53 = 53 in proc_ids
            has_54 = 54 in proc_ids
            print(f"{name}: {proc_ids}")
            if has_53 or has_54:
                print(f"  ✓ Contains processor 53: {has_53}, 54: {has_54}")
    except FileNotFoundError:
        print(f"{name}: NOT FOUND")

print("\nSolution used processors: [53, 54]")
