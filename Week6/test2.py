from akapriori import Apriori

dataset = []

rules = apriori(dataset, support=0.05, confidence=0.3, lift=2)

rules_sorted = sorted(rules, key=lambada
x: [x[4], x[3], x[2]], reverse = True)
