import matplotlib.pyplot as plt

rewards = []
while True:
    string = input()
    if string == "end":
        break
    string = string.split(",")[0]
    # replace "mean: " to "" and convert to float
    string = float(string.replace("mean: ", ""))
    rewards.append(string)

# 折れ線グラフ
plt.plot(rewards)
plt.show()
