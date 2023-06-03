from hopfield import put_noise
from hopfield import calculate_accuracy
from hopfield import calculate_similarity
from hopfield import network_loop
from hopfield import calculate_weight
import numpy as np
import matplotlib.pyplot as plt

##アルファベットのS
memory_s = np.array([
    [1,1,1,1,1],
    [1,-1,-1,-1,-1],
    [1,1,1,1,1],
    [-1,-1,-1,-1,1],
    [1,1,1,1,1],
])

#
memory_h = np.array([
    [1,-1,-1,-1, 1],
    [1,-1,-1,-1, 1],
    [1, 1, 1, 1, 1],
    [1,-1,-1,-1, 1],
    [1,-1,-1,-1, 1],
])


probs = np.arange(0,105,step=5) ##ノイズは0 ~ 100%で5%刻みで検証
print("probs",probs)
outputs = []
similarities = []
accuracies = []
trial_n = 100

W = calculate_weight([memory_s,memory_h])
for i in range(len(probs)):
    outputs = []
    similarity_total = 0
    for n in range(trial_n):
        X = put_noise(memory_s,probs[i]) ##ノイズ生成
        X = network_loop(X,W) ##想起
        outputs.append(X)
        similarity_total += calculate_similarity(X,memory_s) 
    similarities.append((float)(similarity_total) / trial_n) ##類似度計算
    accuracies.append(calculate_accuracy(outputs,memory_s)) ##正答率計算
    print("Noise ",probs[i],"% Done")

for i in range(21):
    print("Noise ",probs[i],"%, ","SIM :",similarities[i],"ACC :",accuracies[i])

##Noises = ["5%","10%","15%","20%"]

##Graph for Similarity
plt.plot(probs,similarities)
plt.xlabel("Noise")
plt.ylabel("Similarity")
plt.ylim(0.0,1.0)
plt.title("Noise vs Similarity (2 Memories)")
plt.show()

##Graph for Accuracy
plt.plot(probs,accuracies)
plt.xlabel("Noise")
plt.ylabel("Accuracy")
plt.yticks(np.arange(0,100,step=20))
plt.title("Noise vs Accuracy (2 Memories)")
plt.show()