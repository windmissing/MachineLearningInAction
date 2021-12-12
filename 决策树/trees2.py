from collections import Counter
def classifyForOne(tree, x):
    while(tree["demention"] >= 0):
        if x[tree["demention"]] > tree["value"]:
            tree = tree["right"]
        else:
            tree = tree["left"]
    return tree["value"]

def classify(tree, test_X):
    ret = []
    for x in test_X:
        ret.append(classifyForOne(tree, x))
    return np.array(ret)

def entropy_part(p):
    if p == 0:
        return 0
    return -p * np.log(p)

def calculateEntropy(y_train):
    if y_train.shape[0] == 0:
        return 0
    votes = Counter(y_train)
    p1 = votes[0] / y_train.shape[0]
    p2 = votes[1] / y_train.shape[0]
    p3 = votes[2] / y_train.shape[0]
    return entropy_part(p1) + entropy_part(p2) + entropy_part(p3)

def trySplit(X_train, y_train, d, v):
    l_X = X_train[X_train[:,d] <= v]
    l_y = y_train[X_train[:,d] <= v]
    r_X = X_train[X_train[:,d] > v]
    r_y = y_train[X_train[:,d] > v]
    return (l_X,l_y),(r_X,r_y)

def findBestSplit(X_train, y_train):
    demention = X_train.shape[1]
    value = X_train.shape[0]
    best_d = 0
    best_v = X_train[0,0]
    best_l = None
    best_r = None
    best_entropy = -1
    for d in range(demention):
        s = set(X_train[:,d])
        for v in s:
            l, r = trySplit(X_train, y_train, d, v)
            if (l[1].shape[0] == 0 or r[1].shape[0]==0):
                continue
            en = calculateEntropy(l[1]) + calculateEntropy(r[1])
            if best_entropy < 0 or en < best_entropy:
                best_d = d
                best_v = v
                best_entropy = en
                best_l = l
                best_r = r
    return best_d, best_v, best_l, best_r, best_entropy

def train(X_train, y_train):
    tree = {}
    entropy = calculateEntropy(y_train)
    if entropy > 0:
        d, v, l, r, en = findBestSplit(X_train, y_train)
        if en < entropy:
            tree["demention"] = d
            tree["value"] = v
            tree["left"] = train(l[0],l[1])
            tree["right"] = train(r[0],r[1])
        else:
            tree["demention"] = -1
            tree["value"] = Counter(y_train).most_common()[0][0]
    else:
        tree["demention"] = -1
        tree["value"] = y_train[0]
    return tree