from thesis_v2.squad_loader import SquadLoader


loader = SquadLoader()

chapters, dataset = loader.load()


print("=" * 60)
print("SQUAD LOADER TEST")
print("=" * 60)

print("Number of chapters :", len(chapters))
print("Number of questions:", len(dataset))

print()
print("First chapter:")
print("-" * 60)
print(chapters[0][:500])

print()
print("First question:")
print("-" * 60)
print("Question:", dataset[0][0])
print("Answer  :", dataset[0][1])
