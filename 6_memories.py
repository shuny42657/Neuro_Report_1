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

memory_u = np.array([
    [1,-1,-1,-1, 1],
    [1,-1,-1,-1, 1],
    [1,-1,-1,-1, 1],
    [1,-1,-1,-1, 1],
    [1, 1, 1, 1, 1],
])

memory_n = np.array([
    [1,-1,-1,-1,1],
    [1,1,-1,-1,1],
    [1,-1,1,-1,1],
    [1,-1,-1,1,1,],
    [1,-1,-1,-1,1],
])

memory_1 = np.array([
    [-1,1,1,-1,-1],
    [-1,-1,1,-1,-1],
    [-1,-1,1,-1,-1],
    [-1,-1,1,-1,-1],
    [-1,-1,1,-1,-1],
])

memory_2 = np.array([
    [1,1,1,1,1],
    [-1,-1,-1,-1,1],
    [1,1,1,1,1],
    [1,-1,-1,-1,-1],
    [1,1,1,1,1]
])
probs = [5,10,15,20] ##ノイズは5~20%で5%刻みで検証
##probs = [5]
outputs = [[],[],[],[]]
similarities = []
accuracies = []
trial_n = 100

W = calculate_weight([memory_s,memory_h,memory_u,memory_n,memory_1, memory_2])
for i in range(len(probs)):
    similarity_total = 0
    for n in range(trial_n):
        X = put_noise(memory_s,probs[i]) ##ノイズ生成
        X = network_loop(X,W) ##想起
        outputs[i].append(X)
        similarity_total += calculate_similarity(X,memory_s) 
    similarities.append((float)(similarity_total) / trial_n) ##類似度計算
    accuracies.append(calculate_accuracy(outputs[i],memory_s)) ##正答率計算
    print("Noise ",probs[i],"% Done")

for i in range(4):
    print("Noise ",probs[i],"%, ","SIM :",similarities[i],"ACC :",accuracies[i])

Noises = ["5%","10%","15%","20%"]

##Graph for Similarity
plt.bar(Noises,similarities)
plt.xlabel("Noise")
plt.ylabel("Similarity")
plt.ylim(0.5,1.0)
plt.title("Noise vs Similarity (6 Memories)")
plt.show()

##Graph for Accuracy
plt.bar(Noises,accuracies)
plt.xlabel("Noise")
plt.ylabel("Accuracy")
plt.yticks(np.arange(0,100,step=20))
plt.title("Noise vs Accuracy (6 Memories)")
plt.show()