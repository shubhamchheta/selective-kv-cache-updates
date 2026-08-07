import json


INPUT_FILE = "datasets/squad/train-v2.0.json"
OUTPUT_FILE = "datasets/squad/questions.json"


MAX_QUESTIONS = 50


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


questions = []


for article in data["data"]:

    for paragraph in article["paragraphs"]:

        context = paragraph["context"]

        for qa in paragraph["qas"]:

            if qa.get("is_impossible", False):
                continue

            answers = qa.get("answers", [])

            if not answers:
                continue

            question = qa["question"]
            answer = answers[0]["text"]

            questions.append({
                "question": question,
                "answer": answer
            })

            if len(questions) >= MAX_QUESTIONS:
                break

        if len(questions) >= MAX_QUESTIONS:
            break

    if len(questions) >= MAX_QUESTIONS:
        break


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        questions,
        f,
        indent=4,
        ensure_ascii=False
    )


print("Questions created:", len(questions))
