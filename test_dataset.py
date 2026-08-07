import textbook_dataset

chapters, questions = textbook_dataset.load("v1")

print("=" * 50)

print("Number of chapters")

print(len(chapters))

print("=" * 50)

for i, chapter in enumerate(chapters):

    print()

    print("Chapter", i + 1)

    print(chapter[:80])

print()

print("=" * 50)

print("Questions")

print(len(questions))
