import json
import os


TEXTBOOK_DIR = "datasets/squad/small_textbook"
QUESTIONS_FILE = "datasets/squad/questions_small.json"


class SquadLoader:

    def load(self,max_questions=None):

        chapters = []

        # --------------------------------------------------
        # Load textbook chapters
        # --------------------------------------------------

        files = sorted(
            f for f in os.listdir(TEXTBOOK_DIR)
            if f.endswith(".txt")
        )

        for file in files:

            path = os.path.join(
                TEXTBOOK_DIR,
                file
            )

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                chapters.append(f.read())

        # --------------------------------------------------
        # Load evaluation questions
        # --------------------------------------------------

        with open(
            QUESTIONS_FILE,
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

        if max_questions is not None:
            dataset = dataset[:max_questions]

        return chapters, dataset
