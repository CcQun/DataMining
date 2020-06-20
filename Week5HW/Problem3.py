import pandas as pd
import math


# 获取一个DataFrame中某一个feature下的所有类别、每个类别的数量、每个类别的索引
def get_all_classes(dataFrame, feature):
    df_group = dataFrame.groupby(by=feature)  # 按照feature分组
    classes = list(df_group.groups.keys())
    num_classes = []
    groups = []
    for i in classes:
        num_classes.append(len(df_group.get_group(name=i)))
        groups.append(df_group.get_group(name=i).index)
    return classes, num_classes, groups


# 计算某一个DataFrame中某个feature的信息熵
def calculate_info_entropy(dataFrame, feature):
    total_num = len(dataFrame)
    classes, num_classes, groups = get_all_classes(dataFrame, feature)
    info_entropy = 0  # 信息熵初始化为0
    for i in range(len(classes)):
        p = num_classes[i] / total_num
        info_entropy += - p * math.log(p, 2)
    return info_entropy


# 计算feature2关于feature1的条件熵
def calculate_conditional_entropy(dataFrame, feature1, feature2):
    total_num = len(dataFrame)
    classes, num_classes, groups = get_all_classes(dataFrame, feature1)
    conditional_entropy = 0
    for i in range(len(classes)):  # 遍历关于feature1的每个分组
        p = num_classes[i] / total_num
        info_entropy = calculate_info_entropy(dataFrame.loc[groups[i], :], feature2)  # 计算每个分组的信息熵
        conditional_entropy += p * info_entropy  # 将每个分组的信息熵乘以该分组的概率然后加到条件熵里
    return conditional_entropy


# 计算feature2关于feature1的信息增益
def calculate_KL_divergence(dataFrame, feature1, feature2):
    info_entropy = calculate_info_entropy(dataFrame, feature2)
    conditional_entropy = calculate_conditional_entropy(dataFrame, feature1, feature2)
    return info_entropy - conditional_entropy


# 建树
def create_Decision_tree(dataFrame):
    features = dataFrame.columns  # 获取dataFrame的所有特征
    KL_divergences = []

    # 获取dataFrame关于最后一列的信息熵，用于计算信息增益
    info_entropy = calculate_info_entropy(dataFrame, features[-1])
    for i in range(len(features) - 1):  # 计算各特征信息增益
        KL_divergence = calculate_KL_divergence(dataFrame, features[i], features[-1])

        if KL_divergence == info_entropy:  # 如果该特征的条件熵为0说明可以直接当做叶子节点了
            classes, num_classes, groups = get_all_classes(dataFrame, features[i])
            decision_tree = {}
            for k in range(len(groups)):
                decision_tree[features[i] + ' ' + str(classes[k])] = dataFrame.loc[groups[k][0]][-1]
            return decision_tree
        else:
            KL_divergences.append(calculate_KL_divergence(dataFrame, features[i], features[-1]))
    most_gain = KL_divergences.index(max(KL_divergences))  # 获取最大信息增益点的feature
    classes, num_classes, groups = get_all_classes(dataFrame, features[most_gain])  # 获取分组信息
    decision_tree = {}
    for i in range(len(groups)):
        index = [j for j in range(len(features))]
        index.pop(most_gain)
        if calculate_info_entropy(dataFrame.iloc[groups[i], index], features[-1]) == 0:  # 该组条件熵为0就可以当做叶子节点
            decision_tree[features[most_gain] + ' ' + str(classes[i])] = dataFrame.loc[groups[i][0]][-1]
        else:  # 条件熵不为0就递归调用该方法直到找到叶子节点
            decision_tree[features[most_gain] + ' ' + str(classes[i])] = create_Decision_tree(
                dataFrame.iloc[groups[i], index])
    return decision_tree


# 使用决策树进行分类
def judge(decision_tree, data):
    for i in decision_tree.keys():
        if data[i.split()[0]] == i.split()[1]:
            if type(decision_tree[i]) == str:
                return decision_tree[i]
            else:
                return judge(decision_tree[i], data)


df = pd.read_csv('p3.csv', error_bad_lines=False)

features = df.columns

print('==========决策树==========')
df = pd.read_csv('p3.csv', error_bad_lines=False)
print(create_Decision_tree(df))
print('==========决策树==========')
