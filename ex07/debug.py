# Load and inspect the file
with open('resources/tweets_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print(f"First 5 lines:")
for i, line in enumerate(lines[:5]):
    print(f"Line {i}: {repr(line)}")
