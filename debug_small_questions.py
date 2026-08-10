import json
import os


SQUAD_FILE = "datasets/squad/train-v2.0.json"
TEXTBOOK_DIR = "datasets/squad/small_textbook"


def normalize(text):
    return " ".join(
        text.lower()
        .replace("\n", " ")
        .split()
    )


# ------------------------------------------------------------
# Load small textbook
# ------------------------------------------------------------

chapters = {}

for i in range(1, 9):

    path = os.path.join(
        TEXTBOOK_DIR,
        f"chapter_{i:02d}.txt"
    )

    with open(path, "r", encoding="utf-8") as f:
        chapters[i] = f.read()

normalized_chapters = {
    i: normalize(text)
    for i, text in chapters.items()
}


# ------------------------------------------------------------
# Load SQuAD
# ------------------------------------------------------------

with open(SQUAD_FILE, "r", encoding="utf-8") as f:
    squad = json.load(f)


# ------------------------------------------------------------
# Find answers inside small textbook
# ------------------------------------------------------------

print("=" * 60)
print("DEBUGGING SMALL TEXTBOOK QUESTIONS")
print("=" * 60)

total_answers = 0

matches = {
    i: []
    for i in range(1, 9)
}


for article in squad["data"]:

    for paragraph in article["paragraphs"]:

        for qa in paragraph["qas"]:

            if qa.get("is_impossible", False):
                continue

            answers = qa.get("answers", [])

            if not answers:
                continue

            answer = answers[0]["text"]

            answer_normalized = normalize(answer)

            for chapter_number, chapter_text in normalized_chapters.items():

                if answer_normalized in chapter_text:

                    matches[chapter_number].append({

                        "question": qa["question"],

                        "answer": answer,

                        "context": paragraph["context"]

                    })


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

for chapter_number in range(1, 9):

    print()
    print("-" * 60)

    print(
        f"Chapter {chapter_number:02d}: "
        f"{len(matches[chapter_number])} answer matches"
    )

    for item in matches[chapter_number][:5]:

        print()
        print("Question:")
        print(item["question"])

        print("Answer:")
        print(item["answer"])

        print("Original context:")
        print(item["context"][:300], "...")


print()
print("=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
