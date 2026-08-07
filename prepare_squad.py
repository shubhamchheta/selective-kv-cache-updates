import json
import os


INPUT_FILE = "datasets/squad/raw/dev-v2.0.json"

TEXTBOOK_DIR = "datasets/squad/textbook"
QUESTIONS_FILE = "datasets/squad/questions.json"


def main():

    os.makedirs(TEXTBOOK_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []

    chapter_id = 1

    for article in data["data"]:

        title = article["title"]

        paragraphs = []

        for paragraph in article["paragraphs"]:

            context = paragraph["context"]

            paragraphs.append(context)

            for qa in paragraph["qas"]:

                if qa.get("is_impossible", False):
                    continue

                if len(qa["answers"]) == 0:
                    continue

                question = qa["question"]

                answer = qa["answers"][0]["text"]

                questions.append({
                    "question": question,
                    "answer": answer,
                    "chapter": chapter_id,
                    "title": title
                })

        chapter_text = f"Chapter {chapter_id}: {title}\n\n"

        chapter_text += "\n\n".join(paragraphs)

        output_file = os.path.join(
            TEXTBOOK_DIR,
            f"chapter_{chapter_id:02d}.txt"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(chapter_text)

        print(
            f"Chapter {chapter_id:02d}: "
            f"{title} | "
            f"{len(paragraphs)} paragraphs"
        )

        chapter_id += 1

    with open(
        QUESTIONS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            questions,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("SQuAD PREPARATION COMPLETE")
    print("=" * 60)

    print("Chapters :", chapter_id - 1)
    print("Questions:", len(questions))


if __name__ == "__main__":
    main()