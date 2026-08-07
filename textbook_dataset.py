import json
import os


def load(version="v1"):

    if version == "v1":
        folder = "./datasets/textbook/version1"
    else:
        folder = "./datasets/textbook/version2"

    chapters = []

    files = sorted(os.listdir(folder))

    for file in files:

        if file.endswith(".txt"):

            path = os.path.join(folder, file)

            with open(path, "r", encoding="utf-8") as f:

                chapters.append(f.read())

    with open(
        "./datasets/textbook/questions.json",
        "r",
        encoding="utf-8"
    ) as f:

        qas = json.load(f)

    dataset = []

    for item in qas:

        dataset.append(
            (
                item["question"],
                item["answer"]
            )
        )

    return chapters, dataset
