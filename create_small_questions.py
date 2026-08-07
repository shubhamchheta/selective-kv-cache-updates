import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

SQUAD_FILE = "datasets/squad/train-v2.0.json"

TEXTBOOK_DIR = "datasets/squad/textbook"

OUTPUT_FILE = "datasets/squad/questions_small.json"

NUM_CHAPTERS = 8
QUESTIONS_PER_CHAPTER = 5


# ============================================================
# LOAD SQUAD
# ============================================================

print("Loading SQuAD dataset...")

with open(SQUAD_FILE, "r", encoding="utf-8") as f:
    squad = json.load(f)


# ============================================================
# LOAD SMALL TEXTBOOK
# ============================================================

chapters = {}

print("\nLoading small textbook...")

for i in range(1, NUM_CHAPTERS + 1):

    path = os.path.join(
        TEXTBOOK_DIR,
        f"chapter_{i:02d}.txt"
    )

    with open(path, "r", encoding="utf-8") as f:
        chapters[i] = f.read()

    print(
        f"Chapter {i:02d}: "
        f"{len(chapters[i].split())} words"
    )


# ============================================================
# FIND QUESTIONS
# ============================================================

chapter_questions = {
    i: []
    for i in range(1, NUM_CHAPTERS + 1)
}


print("\nFinding questions belonging to small chapters...")


for article in squad["data"]:

    for paragraph in article["paragraphs"]:

        context = paragraph["context"]

        # ----------------------------------------------------
        # Check each question
        # ----------------------------------------------------

        for qa in paragraph["qas"]:

            if qa.get("is_impossible", False):
                continue

            answers = qa.get("answers", [])

            if not answers:
                continue

            question = qa["question"]

            answer = answers[0]["text"]

            # ------------------------------------------------
            # Check which small chapter contains the answer
            # ------------------------------------------------

            for chapter_number, chapter_text in chapters.items():

                chapter_lower = chapter_text.lower()
                answer_lower = answer.lower()

                if answer_lower in chapter_lower:

                    # Avoid duplicates
                    duplicate = any(
                        q["question"] == question
                        for q in chapter_questions[chapter_number]
                    )

                    if not duplicate:

                        chapter_questions[chapter_number].append({

                            "question": question,

                            "answer": answer,

                            "context": context

                        })

                    break


# ============================================================
# SELECT QUESTIONS
# ============================================================

questions = []

print("\nSelecting questions...")


for chapter_number in range(1, NUM_CHAPTERS + 1):

    available = chapter_questions[chapter_number]

    selected = available[:QUESTIONS_PER_CHAPTER]

    print(
        f"Chapter {chapter_number:02d}: "
        f"{len(available)} available -> "
        f"{len(selected)} selected"
    )

    for item in selected:

        questions.append({

            "chapter": chapter_number,

            "question": item["question"],

            "answer": item["answer"]

        })


# ============================================================
# SAVE
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        questions,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("QUESTION GENERATION COMPLETE")
print("=" * 60)

print("Total questions:", len(questions))

for chapter_number in range(1, NUM_CHAPTERS + 1):

    count = sum(
        1
        for q in questions
        if q["chapter"] == chapter_number
    )

    print(
        f"Chapter {chapter_number:02d}: "
        f"{count} questions"
    )

print("\nSaved to:")
print(OUTPUT_FILE)