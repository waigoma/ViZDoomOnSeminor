import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

is_kd = False
is_total_reward = False
is_reward_mean = True


def extract_result():
    lines = []
    while True:
        line = input()
        if line == "end":
            break
        if is_kd:
            if "frag:" in line:
                line = line.split("frag: ")[1]
                lines.append(line)
            if "death:" in line:
                line = line.split("death: ")[1]
                lines.append(line)
        if is_total_reward:
            if "total_reward:" in line:
                line = line.split(",")
                if is_reward_mean:
                    line = line[1].split("step_mean: ")[1]
                else:
                    line = line[0].split("total_reward: ")[1]
                lines.append(line)


    results = [[], []]
    i = 0
    # 結果を表示する
    if is_kd:
        for _ in range(int(len(lines)/2)):
            results[0].append(lines[i])
            results[1].append(lines[i+1])
            i += 2
    if is_total_reward:
        results[0].append(lines[0])

rewards = []
kills = []
deaths = []
while True:
    string = input()
    if string == "end":
        break
    if "Epoch" in string:
        continue

    # string = string.split(",")[1]
    # replace "mean: " to "" and convert to float
    # string = float(string.replace("mean: ", "").split(" ")[0])
    # string = float(string.replace("total: ", ""))
    # rewards.append(string)
    kills.append(float(string.split(" ")[0]))
    deaths.append(float(string.split(" ")[1]))

# 折れ線グラフ
# plt.plot(rewards)
plt.plot(kills)
plt.plot(deaths)
plt.show()
