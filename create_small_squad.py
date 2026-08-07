import os

SOURCE_DIR = "datasets/squad/textbook"
OUTPUT_DIR = "datasets/squad/small_textbook"

MAX_WORDS = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(1, 11):

    source_path = os.path.join(
        SOURCE_DIR,
        f"chapter_{i:02d}.txt"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"chapter_{i:02d}.txt"
    )

    print(f"Processing Chapter {i}")

    with open(source_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()

    small_text = " ".join(words[:MAX_WORDS])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(small_text)

    print(f"Original words : {len(words)}")
    print(f"New words      : {len(small_text.split())}")
    print("-" * 40)

print("Done.")