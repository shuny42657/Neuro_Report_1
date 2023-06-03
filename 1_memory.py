from hopfield import put_noise
from hopfield import calculate_accuracy
from hopfield import calculate_similarity
from hopfield import network_loop
from hopfield import calculate_weight
import numpy as np
import matplotlib.pyplot as plt

##アルファベットのS
memory_2 = np.array([
    [1,1,1,1,1],
    [1,-1,-1,-1,-1],
    [1,1,1,1,1],
    [-1,-1,-1,-1,1],
    [1,1,1,1,1],
])
probs = [5,10,15,20] ##ノイズは5~20%で5%刻みで検証
outputs = [[],[],[],[]]
similarities = []
accuracies = []
trial_n = 100

W = calculate_weight([memory_2])
for i in range(len(probs)):
    similarity_total = 0
    for n in range(trial_n):
        X = put_noise(memory_2,probs[i]) ##ノイズ生成
        X = network_loop(X,W) ##想起
        outputs[i].append(X)
        similarity_total += calculate_similarity(X,memory_2) 
    similarities.append((float)(similarity_total) / trial_n) ##類似度計算
    accuracies.append(calculate_accuracy(outputs[i],memory_2)) ##正答率計算
    print("Noise ",probs[i],"% Done")

for i in range(4):
    print("Noise ",probs[i],"%, ","SIM :",similarities[i],"ACC :",accuracies[i])

Noises = ["5%","10%","15%","20%"]

##Graph for Similarity
fig_sim = plt.bar(Noises,similarities)
plt.xlabel("Noise")
plt.ylabel("Similarity")
##plt.yticks(np.arange(0.9,1.0,step=100))
plt.ylim(0.95,1.0)
plt.title("Noise vs Similarity (1 Memory)")
plt.show()
plt.savefig("Noise_vs_Similarity_1_memory")

##Graph for Accuracy
plt.bar(Noises,accuracies)
plt.xlabel("Noise")
plt.ylabel("Accuracy")
plt.yticks(np.arange(0,101,step=20))
plt.title("Noise vs Accuracy (1 Memory)")
plt.show()
plt.savefig("Noise_vs_Accuracy_1_memory")


