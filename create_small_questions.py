import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

SQUAD_FILE = "datasets/squad/train-v2.0.json"

TEXTBOOK_DIR = "datasets/squad/small_textbook"

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
        text = f.read()

    chapters[i] = text

    print(
        f"Chapter {i:02d}: "
        f"{len(text.split())} words"
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    """
    Normalize text for robust matching.
    """

    return " ".join(
        text.lower()
        .replace("\n", " ")
        .split()
    )


normalized_chapters = {
    chapter: normalize(text)
    for chapter, text in chapters.items()
}


# ============================================================
# FIND QUESTIONS
# ============================================================

chapter_questions = {
    i: []
    for i in range(1, NUM_CHAPTERS + 1)
}


print("\nFinding questions that are answerable "
      "from the SMALL textbook...")


for article in squad["data"]:

    for paragraph in article["paragraphs"]:

        context = paragraph["context"]

        context_normalized = normalize(context)

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We only consider a SQuAD context if the COMPLETE
        # context occurs inside our small textbook.
        #
        # This guarantees that the question belongs to the
        # actual knowledge available to our experiment.
        # ----------------------------------------------------

        matching_chapters = []

        for chapter_number, chapter_text in normalized_chapters.items():

            if context_normalized in chapter_text:

                matching_chapters.append(chapter_number)

        # Context does not exist in small textbook
        if not matching_chapters:
            continue


        # ----------------------------------------------------
        # Process questions
        # ----------------------------------------------------

        for qa in paragraph["qas"]:

            # Skip impossible questions
            if qa.get("is_impossible", False):
                continue

            answers = qa.get("answers", [])

            if not answers:
                continue

            question = qa["question"]

            answer = answers[0]["text"]

            answer_normalized = normalize(answer)


            # ------------------------------------------------
            # Make sure answer is actually present
            # ------------------------------------------------

            for chapter_number in matching_chapters:

                if answer_normalized not in normalized_chapters[chapter_number]:
                    continue


                # Avoid duplicate questions
                duplicate = any(
                    q["question"] == question
                    for q in chapter_questions[chapter_number]
                )

                if duplicate:
                    continue


                chapter_questions[chapter_number].append({

                    "question": question,

                    "answer": answer,

                    "context": context

                })


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