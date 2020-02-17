import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

np.random.seed(42)
ACCEPT = "accept"
DECLINE = "decline"
REVIEW = "review"

ACCEPT_PRIORITY = 1
REVIEW_PRIORITY = 2
DECLINE_PRIORITY = 3
REVIEW_PRIORITY_2 = 4
ACCEPT_PRIORITY_2 = 5
ACCEPT_PRIORITY_3 = 6
REVIEW_PRIORITY_3 = 7
DECLINE_PRIORITY_2 = 8
REVIEW_PRIORITY_4 = 9
ACCEPT_PRIORITY_4 = 10

number_transactions = 225000
fraud_rate = 0.05

number_fraud_transactions = int(number_transactions * fraud_rate)
number_legit_transactions = number_transactions - number_fraud_transactions

positive_labels = np.ones(number_fraud_transactions, dtype=int)
negative_labels = np.zeros(number_legit_transactions, dtype=int)

labels = np.concatenate((positive_labels, negative_labels), axis=0)

number_accept_rules = 7
number_review_rules = 30
number_decline_rules = 30

distribution_min = 0
distribution_max = number_transactions


def sampleFromNormal(mean, std):
    sample = -1
    global number_transactions
    while sample < 0 | sample > number_transactions:
        sample = np.random.normal(mean, std, 1)
    return sample


def get_truncated_normal(mean=0, sd=1, low=0, upp=number_transactions):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def plot_Distribution(dist, title, mu, sigma):
    plt.figure()
    s = dist.rvs(1000)
    count, bins, ignored = plt.hist(s, 60, density=True)
    plt.title(title)


did_plots = False

accept_rules_distribution_mean = number_transactions / 5
accept_rules_distribution_std = number_transactions / 10

trunced_accept_support = get_truncated_normal(
    mean=accept_rules_distribution_mean,
    sd=accept_rules_distribution_std,
    low=0,
    upp=number_transactions,
)
plot_Distribution(
    trunced_accept_support,
    "trunced_accept_support",
    accept_rules_distribution_mean,
    accept_rules_distribution_std,
)
trunced_accept_accuracy = get_truncated_normal(mean=3 / 4, sd=1 / 5, low=0, upp=1)
plot_Distribution(trunced_accept_accuracy, "trunced_accept_accuracy", 3 / 4, 1 / 5)


decline_rules_distribution_mean = number_transactions * 0.0001
decline_rules_distribution_std = number_transactions * 0.001

trunced_decline_support = get_truncated_normal(
    mean=decline_rules_distribution_mean,
    sd=decline_rules_distribution_std,
    low=0,
    upp=number_transactions,
)
plot_Distribution(
    trunced_decline_support,
    "trunced_decline_support",
    decline_rules_distribution_mean,
    decline_rules_distribution_std,
)
trunced_decline_accuracy = get_truncated_normal(mean=1 / 6, sd=1 / 20, low=0, upp=1)
plot_Distribution(trunced_decline_accuracy, "trunced_decline_accuracy", 1 / 6, 1 / 20)


class Rule:
    def __init__(self, labels, action=DECLINE):
        number_positive_in_dataset = np.sum(labels == 1)
        number_negative_in_dataset = np.sum(labels == 0)

        diff_negative = -1
        diff_positive = -1

        if action == ACCEPT:
            self.priotity = np.random.choice(
                [
                    ACCEPT_PRIORITY,
                    ACCEPT_PRIORITY_2,
                    ACCEPT_PRIORITY_3,
                    ACCEPT_PRIORITY_4,
                ],
                1,
                p=[0.3, 0.3, 0.2, 0.2],
            )[0]

            while (diff_negative < 0) | (diff_positive < 0):
                self.support_number = int(trunced_accept_support.rvs())

                self.number_correct_triggers = int(
                    trunced_accept_accuracy.rvs() * self.support_number
                )

                number_of_positive_triggers = (
                    self.support_number - self.number_correct_triggers
                )
                number_of_negative_triggers = self.number_correct_triggers
                diff_positive = number_positive_in_dataset - number_of_positive_triggers
                diff_negative = number_negative_in_dataset - number_of_negative_triggers
        else:
            if action == DECLINE:
                self.priotity = np.random.choice(
                    [DECLINE_PRIORITY, DECLINE_PRIORITY_2], 1, p=[0.6, 0.4]
                )[0]
                self.priotity = DECLINE_PRIORITY
            else:
                self.priotity = np.random.choice(
                    [
                        REVIEW_PRIORITY,
                        REVIEW_PRIORITY_2,
                        REVIEW_PRIORITY_3,
                        REVIEW_PRIORITY_4,
                    ],
                    1,
                    p=[0.3, 0.3, 0.2, 0.2],
                )[0]

            while (diff_negative < 0) | (diff_positive < 0):
                self.support_number = int(trunced_decline_support.rvs())

                self.number_correct_triggers = int(
                    trunced_decline_accuracy.rvs() * self.support_number
                )

                number_of_negative_triggers = (
                    self.support_number - self.number_correct_triggers
                )
                number_of_positive_triggers = self.number_correct_triggers
                diff_negative = number_negative_in_dataset - number_of_negative_triggers
                diff_positive = number_positive_in_dataset - number_of_positive_triggers


def generateColumnsRuleTriggersAccept(rule, labels):
    positive_index_labels = np.squeeze(np.where(labels == 1))
    negative_index_labels = np.squeeze(np.where(labels == 0))
    number_negative_triggered = rule.number_correct_triggers
    number_positive_triggered = rule.support_number - number_negative_triggered

    trigered_positive_indexes = np.random.choice(
        positive_index_labels, number_positive_triggered
    )
    trigered_negative_indexes = np.random.choice(
        negative_index_labels, number_negative_triggered
    )

    triggers_vector = np.full(np.shape(labels), -1)
    triggers_vector[trigered_positive_indexes] = rule.priotity
    triggers_vector[trigered_negative_indexes] = rule.priotity

    return triggers_vector


initialized_triggers_matrix = np.full(
    [
        number_transactions,
        number_accept_rules + number_decline_rules + number_review_rules,
    ],
    1,
)

# Generate N accept Rules
list_accept_rules = list()
list_rule_names = list()
for r_A in range(number_accept_rules):
    rule = Rule(labels, action=ACCEPT)

    list_rule_names.append("RULE_accept_" + str(r_A + 1))
    initialized_triggers_matrix[:, r_A] = generateColumnsRuleTriggersAccept(
        rule, labels
    )

for r_D in range(number_decline_rules):
    rule = Rule(labels, action=DECLINE)

    list_rule_names.append("RULE_decline_" + str(r_D + 1))
    initialized_triggers_matrix[:, r_D + r_A + 1] = generateColumnsRuleTriggersAccept(
        rule, labels
    )

for r_R in range(number_review_rules):
    rule = Rule(labels, action=REVIEW)

    list_rule_names.append("RULE_review_" + str(r_R + 1))
    initialized_triggers_matrix[
        :, r_R + r_D + r_A + 1 + 1
    ] = generateColumnsRuleTriggersAccept(rule, labels)

dataframe = pd.DataFrame.from_records(initialized_triggers_matrix)


data_frame_reviews = dataframe.iloc[
    :, number_decline_rules + number_accept_rules :
].copy()

for i in range(data_frame_reviews.shape[1]):
    old_col = data_frame_reviews.iloc[:, i].to_numpy()
    indexes = np.argwhere(old_col > -1)
    np.random.shuffle(indexes)
    old_col[:] = -1
    indexes = indexes[0 : int(0.9 * np.shape(indexes)[0]), :]
    old_col[indexes] = DECLINE_PRIORITY
    list_rule_names.append("RULE_decline_MOD_" + str(i + 1))
    print(i)
result = pd.concat([dataframe, data_frame_reviews], axis=1, sort=False)


def build_fradubyvalidation(label):
    if label == 1:
        return "true"
    else:
        return "confirmedfalse"


result["label"] = np.array(labels, dtype=bool)

result["fraud_by_validation"] = result.apply(
    lambda row: build_fradubyvalidation(row.label), axis=1
)

result.drop(["label"], axis=1, inplace=True)
result["amount"] = 50
result["B_RULE_accept_0"] = ACCEPT_PRIORITY
result.columns = (
    list_rule_names + ["fraud_by_validation"] + ["amount"] + ["B_RULE_accept_0"]
)
result = result[
    ["B_RULE_accept_0"] + list_rule_names + ["fraud_by_validation"] + ["amount"]
]


result = result.sample(frac=1).reset_index(drop=True)
result["amount"] = result["amount"].round(decimals=2)
result["amount"] = result["amount"].map("{:,.2f}".format)

dataframe_train = result.iloc[0:75000, :]
dataframe_val = result.iloc[75000:150000, :]
dataframe_test = result.iloc[150000:, :]

dataframe_train.to_csv(
    "sintetic_data_complexprios_train_decline_MOD.csv", sep=",", index=False
)
dataframe_val.to_csv(
    "sintetic_data_complexprios_val_decline_MOD.csv", sep=",", index=False
)
dataframe_test.to_csv(
    "sintetic_data_complexprios_test_decline_MOD.csv", sep=",", index=False
)
