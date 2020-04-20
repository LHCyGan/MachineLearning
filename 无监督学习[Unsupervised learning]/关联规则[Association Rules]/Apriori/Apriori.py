from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """其中C1即为元素个数为1的项集（非频繁项集，因为还没有同最小支持度比较）。
    map(frozenset, C1)的语义是将C1由Python列表转换为不变集合（frozenset，Python中的数据结构）。"""
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # 这里必须要使用frozenset而不是set类型，因为之后 必须要将这些集合作为字典键值使用，使用frozenset可以实现这一点，而set却做不到
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    """其中D为全部数据集，Ck为大小为k（包含k个元素）的候选项集，minSupport为设定的最小支持度。
    返回值中retList为在Ck中找出的频繁项集（支持度大于minSupport的），supportData记录各频繁项集的支持度。
    retList.insert(0, key)一行将频繁项集插入返回列表的首部。"""

    # 创建一个空字典ssCnt
    ssCnt = {}
    """
    遍历数据集中的 所有交易记录以及C1中的所有候选集。如果C1中的集合是记录的一部分，那么增加字典中对应 的计数值。这里字典的键就是集合
    """
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    # 构建一个空列表，该列表包含满足 最小支持度要求的集合
    retList = []
    supportData = {}
    # 计算支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        # 在列表的首部插入任意新的集合。当然也不一定非要在首部插入，这只是为了让列表看起来有组织.
        if support >= minSupport:
            retList.insert(0, key)
        # 返回最频繁项集的支持度supportData
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """的输入参数为频繁项集列表Lk与项集元素个数k，输出为Ck"""
    # TODO: 举例来说， 该函数以{0}、{1}、{2}作为输入，会生成{0,1}、{0,2}以及{1,2}。
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 前k-2项相同时，将两个集合合并
            """
            上面的k-2有点让人疑惑。接下来再进一步讨论细节。当利用{0}、{1}、 {2}构建{0,1}、{0,2}、{1,2}时，这实际上是将单个项组合到一块。
            现在如果想利用{0,1}、 {0,2}、 {1,2}来创建三元素 项集，应该怎么做？如果将每两个集合合并，就会得到{0, 1, 2}、 {0, 1, 2}、 {0, 1, 2}。也就是说， 
            同样的结果集合会重复3次。接下来需要扫描三元素项集列表来得到非重复结果，我们要做的是 确保遍历列表的次数最少。现在，如果比较集合{0,1}、 {0,2}、 {1,2}
            的第1个元素并只对第1个元 素相同的集合求并操作，又会得到什么结果？{0, 1, 2}，而且只有一次操作！这样就不需要遍历 列表来寻找非重复值
            """
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                # | 为集合的求并操作
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    """给该函数传递一个数据集以及一个支持度， 函数会生成候选项集的列表"""
    # 首先创建C1然后读入数据集将其转化为D（集合列表）来完 成
    # C1包含了每个frozenset中的单个物品项：C1是一个集合的集合，如{{0},{1},{2},…}，每次添加的都是单个项构成的集合{0}、{1}、{2}
    C1 = createC1(dataSet)
    # 中使用map函数将set()映射到dataSet列表中的每一项
    # 集合形式的数据
    D = map(set, dataSet)
    # 去掉那些不满足最小支持度的项集
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    """
    首先使 用aprioriGen()来创建Ck，然后使用scanD()基于Ck来创建Lk。
    Ck是一个候选项集列表，然 后scanD()会遍历Ck，丢掉不满足最小支持度要求的项集 。
    Lk列表被添加到L，同时增加k的 值，重复上述过程。最后，当Lk为空时，程序返回L并退出
    """

    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        # 扫描数据集，从Ck得到Lk
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData