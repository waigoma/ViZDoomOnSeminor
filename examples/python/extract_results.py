# 複数行にわたるテキストから、"Results", "Epoch" が含まれる行のみを抽出する
# Results: の前に文字列がある場合は消去する
# 終わりの行には end を入力する

results = []
while True:
    line = input()
    if line == "end":
        break
    if "Results" in line:
        line = line.split("Results: ")[1]
        results.append(line)
    if "Epoch" in line:
        results.append(line)

# 結果を表示する
for result in results:
    print(result)
