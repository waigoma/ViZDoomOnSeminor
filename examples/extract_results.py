# 複数行にわたるテキストから、"Results", "Epoch" が含まれる行のみを抽出する
# Results: の前に文字列がある場合は消去する
# 終わりの行には end を入力する

is_kd = False
is_result = True
results = []
while True:
    line = input()
    if line == "end":
        break
    if is_kd:
        if "frag:" in line:
            line = line.split("frag: ")[1]
            results.append(line)
        if "death:" in line:
            line = line.split("death: ")[1]
            results.append(line)
    if is_result:
        if "total_reward:" in line:
            line = line.split(",")[0].split("total_reward: ")[1]
            results.append(line)
        if "Epoch" in line:
            results.append(line)

i = 0
# 結果を表示する
if is_kd:
    for _ in range(int(len(results)/2)):
        print(results[i], results[i+1])
        i += 2
if is_result:
    for result in results:
        print(result)
