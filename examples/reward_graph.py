import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')

rewards = []
kills = []
deaths = []
while True:
    string = input()
    if string == "end":
        break
    if "Epoch" in string:
        continue

    # replace "mean: " to "" and convert to float
    # string = float(string.split(",")[0].replace("mean: ", "").split(" ")[0])
    string = float(string.split(",")[1].replace("total: ", ""))
    # string = float(string)
    rewards.append(string)
    # kills.append(float(string.split(" ")[0]))
    # deaths.append(float(string.split(" ")[1]))

# 折れ線グラフ
plt.plot(rewards)
# plt.plot(kills)
# plt.plot(deaths)
plt.show()
