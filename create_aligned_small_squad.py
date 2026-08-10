import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "datasets/squad/train-v2.0.json"

OUTPUT_DIR = "datasets/squad/small_textbook"

QUESTIONS_OUTPUT = "datasets/squad/questions_small.json"

NUM_CHAPTERS = 8

TARGET_WORDS = 400


# ============================================================
# LOAD SQUAD
# ============================================================

print("=" * 60)
print("CREATING ALIGNED SMALL SQUAD DATASET")
print("=" * 60)

print("\nLoading SQuAD...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# EXTRACT COMPLETE CONTEXTS
# ============================================================

contexts = []


for article in data["data"]:

    for paragraph in article["paragraphs"]:

        context = paragraph["context"].strip()

        if not context:
            continue

        qas = []

        for qa in paragraph["qas"]:

            # Skip unanswerable questions
            if qa.get("is_impossible", False):
                continue

            answers = qa.get("answers", [])

            if not answers:
                continue

            qas.append({
                "question": qa["question"],
                "answer": answers[0]["text"]
            })

        # We only keep contexts that have
        # at least one answerable question.
        if qas:

            contexts.append({
                "context": context,
                "questions": qas
            })


print("Complete contexts found:", len(contexts))


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# BUILD SMALL CHAPTERS
# ============================================================

chapters = []

current_contexts = []
current_words = 0

chapter_number = 1


for item in contexts:

    context = item["context"]

    context_words = len(context.split())


    # --------------------------------------------------------
    # If this context alone is larger than target,
    # skip it for now.
    # --------------------------------------------------------

    if context_words > TARGET_WORDS:

        continue


    # --------------------------------------------------------
    # Would adding this context exceed target?
    # --------------------------------------------------------

    if (
        current_words > 0
        and current_words + context_words > TARGET_WORDS
    ):

        chapters.append({
            "contexts": current_contexts,
            "word_count": current_words
        })

        current_contexts = []
        current_words = 0

        chapter_number += 1


    # --------------------------------------------------------
    # Add complete context
    # --------------------------------------------------------

    current_contexts.append(item)

    current_words += context_words


    # Stop once we have enough chapters
    if len(chapters) >= NUM_CHAPTERS:

        break


# Add final chapter if necessary
if (
    len(chapters) < NUM_CHAPTERS
    and current_contexts
):

    chapters.append({
        "contexts": current_contexts,
        "word_count": current_words
    })


# Keep only requested number
chapters = chapters[:NUM_CHAPTERS]


# ============================================================
# SAVE CHAPTERS + QUESTIONS
# ============================================================

all_questions = []


print("\nCreating chapters...")
print("-" * 60)


for chapter_index, chapter in enumerate(
    chapters,
    start=1
):

    chapter_text_parts = []

    chapter_questions = []


    for item in chapter["contexts"]:

        # Keep the COMPLETE context
        chapter_text_parts.append(
            item["context"]
        )


        # Questions belong to this exact context
        for q in item["questions"]:

            chapter_questions.append({

                "chapter": chapter_index,

                "question": q["question"],

                "answer": q["answer"]

            })


    chapter_text = "\n\n".join(
        chapter_text_parts
    )


    # --------------------------------------------------------
    # Save textbook chapter
    # --------------------------------------------------------

    chapter_path = os.path.join(
        OUTPUT_DIR,
        f"chapter_{chapter_index:02d}.txt"
    )


    with open(
        chapter_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(chapter_text)


    # --------------------------------------------------------
    # Add questions
    # --------------------------------------------------------

    all_questions.extend(
        chapter_questions
    )


    print(
        f"Chapter {chapter_index:02d} : "
        f"{chapter['word_count']} words | "
        f"{len(chapter['contexts'])} contexts | "
        f"{len(chapter_questions)} questions"
    )


# ============================================================
# LIMIT QUESTIONS
# ============================================================

# Keep a reasonable number of questions per chapter.

MAX_QUESTIONS_PER_CHAPTER = 5

selected_questions = []


for chapter_index in range(
    1,
    len(chapters) + 1
):

    chapter_questions = [

        q for q in all_questions

        if q["chapter"] == chapter_index

    ]


    selected_questions.extend(
        chapter_questions[
            :MAX_QUESTIONS_PER_CHAPTER
        ]
    )


# ============================================================
# SAVE QUESTIONS
# ============================================================

with open(
    QUESTIONS_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        selected_questions,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET CREATION COMPLETE")
print("=" * 60)

print(
    "Chapters created :",
    len(chapters)
)

print(
    "Questions created:",
    len(selected_questions)
)

print("\nTextbook:")
print(OUTPUT_DIR)

print("\nQuestions:")
print(QUESTIONS_OUTPUT)

print("\nChapter question distribution:")
print("-" * 60)


for chapter_index in range(
    1,
    len(chapters) + 1
):

    count = sum(
        1
        for q in selected_questions
        if q["chapter"] == chapter_index
    )

    print(
        f"Chapter {chapter_index:02d}: "
        f"{count} questions"
    )
