'''
Remove one of the file attributes form the name.
'''
import os

folder = input("Folder: ").strip().strip('"')
index = int(input("Index of the '.'-separated piece to remove: "))

renames = []
for f in sorted(os.listdir(folder)):
    parts = f.split(".")
    if not os.path.isfile(os.path.join(folder, f)) or index >= len(parts):
        continue
    new = ".".join(parts[:index] + parts[index + 1:])
    renames.append((f, new))

for old, new in renames:
    print(f"{old} -> {new}")

if input(f"Rename {len(renames)} files? [y/N]: ").strip().lower() == "y":
    for old, new in renames:
        dst = os.path.join(folder, new)
        if os.path.exists(dst):
            print(f"Skipped {old}: {new} already exists")
            continue
        os.rename(os.path.join(folder, old), dst)
    print("Done")
