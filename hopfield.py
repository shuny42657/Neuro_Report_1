import numpy as np

##重み計算
def calculate_weight(memory = []):
    W = np.zeros((25,25),dtype = np.int8)
    for i in range(25):
        for j in range(25):
            if i == j:
                continue
            else:
                total = 0
                for m in memory:
                    m_ = m.reshape(-1)
                    total += m_[i] * m_[j]
                W[i][j] = total
    return W
##ヘブ則
def hebb(X,W):
    X = X.reshape(-1)
    next_i = np.random.randint(0,24)
    next_X = X.copy()

    total = 0
    for j in range(25):
        total += W[next_i,j] * X[j]
    if total > 0:
        next_X[next_i] = 1
    elif total == 0:
        next_X[next_i] = 0
    elif total < 0:
        next_X[next_i] = -1
        

    ###
    return next_X.reshape(5,5)

##結果表示用関数
def show_net(X : np.ndarray):
    X_ = X.reshape(5,5)
    print(X_)

##ノイズ生成用関数
def put_noise(X,prob): ##probの確率で各ノードの値を反転させる
    X_ = X.copy()
    for i in range(5):
        for j in range(5):
            rand = np.random.randint(1,100)
            if rand <= prob:
                X_[i,j] *= -1
    return X_

##エネルギーの計算（閾値はなし）
def calculate_energy(X,W):
    X_ = X.reshape(-1)
    total = 0
    for i in range(25):
        for j in range(25):
            total += X_[i] * X_[j] * W[i,j]

    return -0.5 * total
##類似度の計算
def calculate_similarity(output,answer):
    output_  = output.reshape(-1)
    answer_ = answer.reshape(-1)
    total = 0
    for i in range(25):
        total += output_[i] * answer_[i]
    return (float)(total) / len(output_)

def determine_similarity(output,answers):
    similarities = []
    for a in answers:
        similarities.append(calculate_similarity(output,a))
    return max(similarities)

##正答率の計算
def calculate_accuracy(outputs,answer):
    answer_ = answer.reshape(-1)
    count = 0
    for o in outputs:
        if(calculate_similarity(o,answer) == 1.0):
            count += 1
    return 100.0 * count / len(outputs)

def network_loop(X,W):
    ##W = calculate_weight(memory)
    old_energy = 0
    count = 0
    Iter = 0
    while(True):
        Iter += 1
        X = hebb(X,W)
        energy = calculate_energy(X,W)
        ##print("Energy :",energy)
        if(energy < old_energy):
            ##print("Energy Descend to ",energy)
            count = 0
        else:
            count += 1
        old_energy = energy
        if count >= 150:
            break
    return X
X = np.array([
    [-1,-1,-1,-1,1],
    [-1,-1,1,-1,-1],
    [-1,1,-1,-1,1],
    [1,-1,1,1,-1],
    [1,1,1,-1,1],
])

memory_ = X.copy()
memory_1 = np.array([
    [1,-1,-1,1,-1],
    [1,-1,-1,1,-1],
    [1,-1,-1,1,-1],
    [1,1,1,1,1],
    [-1,-1,-1,1,-1],
])
memory_2 = np.array([
    [1,1,1,1,1],
    [1,-1,-1,-1,-1],
    [1,1,1,1,1],
    [-1,-1,-1,-1,1],
    [1,1,1,1,1],
])

