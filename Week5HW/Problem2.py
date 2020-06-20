import pandas as pd
import math

# 获取一个DataFrame中某一个feature下的所有类别、每个类别的数量、每个类别的索引
def get_all_classes(dataFrame, feature):
    df_group = dataFrame.groupby(by=feature)
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
    info_entropy = 0
    for i in range(len(classes)):
        p = num_classes[i] / total_num
        info_entropy += - p * math.log(p, 2)
    return info_entropy


# 计算feature2关于feature1的条件熵
def calculate_conditional_entropy(dataFrame, feature1, feature2):
    total_num = len(dataFrame)
    classes, num_classes, groups = get_all_classes(dataFrame, feature1)
    conditional_entropy = 0
    for i in range(len(classes)):
        p = num_classes[i] / total_num
        info_entropy = calculate_info_entropy(dataFrame.loc[groups[i], :], feature2)
        conditional_entropy += p * info_entropy
    return conditional_entropy


# 计算feature2关于feature1的信息增益
def calculate_KL_divergence(dataFrame, feature1, feature2):
    info_entropy = calculate_info_entropy(dataFrame, feature2)
    conditional_entropy = calculate_conditional_entropy(dataFrame, feature1, feature2)
    return info_entropy - conditional_entropy

# 建树
def create_Decision_tree(dataFrame):
    features = dataFrame.columns
    KL_divergences = []
    info_entropy = calculate_info_entropy(dataFrame, features[-1])
    for i in range(len(features) - 1):
        KL_divergence = calculate_KL_divergence(dataFrame, features[i], features[-1])

        if KL_divergence == info_entropy:
            classes, num_classes, groups = get_all_classes(dataFrame, features[i])
            decision_tree = {}
            for k in range(len(groups)):
                decision_tree[features[i] + ' ' + classes[k]] = dataFrame.loc[groups[k][0]][-1]
            return decision_tree
        else:
            KL_divergences.append(calculate_KL_divergence(dataFrame, features[i], features[-1]))
    most_gain = KL_divergences.index(max(KL_divergences))
    classes, num_classes, groups = get_all_classes(dataFrame, features[most_gain])
    decision_tree = {}
    for i in range(len(groups)):
        index = [j for j in range(len(features))]
        index.pop(most_gain)
        if calculate_info_entropy(dataFrame.iloc[groups[i], index], features[-1]) == 0:
            decision_tree[features[most_gain] + ' ' + classes[i]] = dataFrame.loc[groups[i][0]][-1]
        else:
            decision_tree[features[most_gain] + ' ' + classes[i]] = create_Decision_tree(
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


df = pd.read_csv('p2.csv',encoding='gb18030',error_bad_lines=False)



features = df.columns

print('==========条件熵==========')
for i in range(1,len(features) - 1):
    print(features[i],':',calculate_conditional_entropy(df, features[i], features[-1]))
print('==========条件熵==========')
print('==========决策树==========')
print(create_Decision_tree(df.iloc[:,1:]))
print('==========决策树==========')