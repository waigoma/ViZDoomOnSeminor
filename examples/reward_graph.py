import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

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
