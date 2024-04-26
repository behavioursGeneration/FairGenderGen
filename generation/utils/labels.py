import torch
from collections import Counter
import random


# Tableau de labels
categories_dialog_act = ["silence", "Declaration", "Backchannel", "Agree/accept" , "Disagree/disaccept", "Question", "Directive" , "Non-understanding", "Opening", "Apology", "Thanking"]
categories_certainty = ["silence", "Certain", "Neutral", "Uncertainty"]
categories_valence = ["silence", "Positive", "Negative", "Neutral"]
categories_arousal = ["silence", "Active", "Passive", "Neutral"]
categories_dominance = ["silence", "Strong", "Weak", "Neutral"]
categories_gender = ["H", "F", "silence"]
categories_small_gender = ["H", "F"]


# Créer les dictionnaires de mapping
label_to_index_dialog_act  = {label: i for i, label in enumerate(sorted(categories_dialog_act))}
index_to_label_dialog_act  = {i: label for label, i in label_to_index_dialog_act.items()}

label_to_index_valence  = {label: i for i, label in enumerate(sorted(categories_valence))}
index_to_label_valence  = {i: label for label, i in label_to_index_valence.items()}

label_to_index_arousal  = {label: i for i, label in enumerate(sorted(categories_arousal))}
index_to_label_arousal  = {i: label for label, i in label_to_index_arousal.items()}

label_to_index_dominance  = {label: i for i, label in enumerate(sorted(categories_dominance))}
index_to_label_dominance  = {i: label for label, i in label_to_index_dominance.items()}

label_to_index_certainty  = {label: i for i, label in enumerate(sorted(categories_certainty))}
index_to_label_certainty  = {i: label for label, i in label_to_index_certainty.items()}

label_to_index_gender  = {label: i for i, label in enumerate(sorted(categories_gender))}
index_to_label_gender  = {i: label for label, i in label_to_index_gender.items()}

label_to_index_small_gender  = {label: i for i, label in enumerate(sorted(categories_small_gender))}
index_to_label_small_gender  = {i: label for label, i in label_to_index_small_gender.items()}

print("*"*10, "LABELS")
print(label_to_index_gender)
print(index_to_label_gender)

# Fonction pour convertir les labels en représentation one-hot
def label_to_one_hot(label, type):
    if type == "dialog_act":
        label_to_index = label_to_index_dialog_act
    elif type == "valence":
        label_to_index = label_to_index_valence
    elif type == "arousal":
        label_to_index = label_to_index_arousal
    elif type == "dominance":
        label_to_index = label_to_index_dominance
    elif type == "certainty":
        label_to_index = label_to_index_certainty
    elif type == "gender":
        label_to_index = label_to_index_gender
    elif type == "small_gender":
        label_to_index = label_to_index_small_gender
    num_classes = len(label_to_index)
    one_hot = torch.zeros(num_classes)
    one_hot[label_to_index[label]] = 1
    return one_hot

# Fonction pour récupérer le label à partir de la représentation one-hot
def one_hot_to_label(one_hot, type):
    if type == "dialog_act":
        index_to_label = index_to_label_dialog_act
    elif type == "valence":
        index_to_label = index_to_label_valence
    elif type == "arousal":
        index_to_label = index_to_label_arousal
    elif type == "dominance":
        index_to_label = index_to_label_dominance
    elif type == "certainty":
        index_to_label = index_to_label_certainty
    elif type == "gender":
        index_to_label = index_to_label_gender
    elif type == "small_gender":
        index_to_label = index_to_label_small_gender
    index = torch.argmax(one_hot)
    return index_to_label[index.item()]

def one_hot_to_index(one_hot, type):
    index = torch.argmax(one_hot)
    if type == "dialog_act":
        index_to_label = index_to_label_dialog_act
        return label_to_index_dialog_act[index_to_label[index.item()]]
    elif type == "valence":
        index_to_label = index_to_label_valence
        return label_to_index_valence[index_to_label[index.item()]]
    elif type == "arousal":
        index_to_label = index_to_label_arousal
        return label_to_index_arousal[index_to_label[index.item()]]
    elif type == "dominance":
        index_to_label = index_to_label_dominance
        return label_to_index_dominance[index_to_label[index.item()]]
    elif type == "certainty":
        index_to_label = index_to_label_certainty
        return label_to_index_certainty[index_to_label[index.item()]]
    elif type == "gender":
        index_to_label = index_to_label_gender
        return label_to_index_gender[index_to_label[index.item()]]
    elif type == "small_gender":
        index_to_label = index_to_label_small_gender
        return label_to_index_small_gender[index_to_label[index.item()]]

def other_label(categories, current_label_list, type):
    new_list = []
    for one_hot_current_label in current_label_list:
        current_label = one_hot_to_label(one_hot_current_label, type)
        list_wt_label = [l for l in categories if l != current_label]
        new_label = random.choice(list_wt_label)
        new_list.append(label_to_one_hot(new_label, type))
    return torch.stack(new_list)

def get_other_label(label_list, type):
    if type == "dialog_act":
        return other_label(categories_dialog_act, label_list, type)
    elif type == "valence":
        return other_label(categories_valence, label_list, type)
    elif type == "arousal":
        return other_label(categories_arousal, label_list, type)
    elif type == "dominance":
        return other_label(categories_dominance, label_list, type)
    elif type == "certainty":
        return other_label(categories_certainty, label_list, type)
    elif type == "gender":
        return other_label(categories_gender, label_list, type)
    elif type == "small_gender":
        return other_label(categories_small_gender, label_list, type)


def get_labels(type):
    if type == "dialog_act":
        labels = categories_dialog_act
    elif type == "valence":
        labels = categories_valence
    elif type == "arousal":
        labels = categories_arousal
    elif type == "dominance":
        labels = categories_dominance
    elif type == "certainty":
        labels = categories_certainty
    elif type == "gender":
        labels = categories_gender
    elif type == "small_gender":
        labels = categories_small_gender
    return labels

def get_color(type):
    if type == "dialog_act":
        color = {"silence": "grey", "Declaration": "blue", "Backchannel":"yellow", "Agree/accept":"green" , "Disagree/disaccept":"red", "Question":"purple", "Directive":"black" , "Non-understanding":"orange", "Opening":"pink", "Apology":"brown", "Thanking":"olive"}
    elif type == "valence":
        color = {"silence": "grey", "Positive": "green", "Negative":"red", "Neutral":"blue"}
    elif type == "arousal":
        color = {"silence": "grey", "Active": "green", "Passive":"red", "Neutral":"blue"}
    elif type == "dominance":
        color = {"silence": "grey", "Strong": "green", "Weak":"red", "Neutral":"blue"}
    elif type == "certainty":
        color = {"silence": "grey", "Certain": "green", "Neutral":"red", "Uncertainty":"blue"}
    elif type == "gender":
        color = {"silence": "grey", "H": "red", "F" : "green"}
    elif type == "small_gender":
        color = {"H": "red", "F" : "green"}
    return color

def get_labels_to_index(type):
    if type == "dialog_act":
        labels = label_to_index_dialog_act
    elif type == "valence":
        labels = label_to_index_valence
    elif type == "arousal":
        labels = label_to_index_arousal
    elif type == "dominance":
        labels = label_to_index_dominance
    elif type == "certainty":
        labels = label_to_index_certainty
    elif type == "gender":
        labels = label_to_index_gender
    elif type == "small_gender":
        labels = label_to_index_small_gender
    return labels


def get_maj_label(labels):
        # Count the number of occurrences of each label
        label_counts = Counter(labels)
        # Find the majority label
        majority_label = max(label_counts, key=label_counts.get)
        # Calculate the percentage presence of the majority label
        percentage_majority = label_counts[majority_label] / len(labels) * 100
        # If there's something other than silence, we'll take something else.
        if majority_label == "silence" and percentage_majority < 100:
            majority_label = Counter(labels).most_common(2)[1][0]
            #second_percentage_majority = label_counts[second_majority_label] / len(labels) * 100
        return majority_label


def supress_silence_index(data, one_hot_labels_list, type):
    raw_labels_list = [one_hot_to_label(label, type) for label in one_hot_labels_list]
    supress_index = []
    for i in range(len(raw_labels_list)):
        if "silence" in raw_labels_list[i]:
            supress_index.append(i)
    tensor = data.clone()
    masque = torch.ones(data.size(0), dtype=torch.bool).to(tensor)
    masque[supress_index] = False
    tensor_without_silence = torch.index_select(tensor, dim=0, index=torch.nonzero(masque).squeeze()).to(tensor)
    one_hot_labels_without_silence = torch.index_select(one_hot_labels_list, dim=0, index=torch.nonzero(masque).squeeze()).to(tensor)
    labels_without_silence = [raw_labels_list[i] for i in range(len(raw_labels_list)) if i not in supress_index]
    print("******len after supress silence**********")
    print(len(tensor_without_silence), len(labels_without_silence))
    return tensor_without_silence, one_hot_labels_without_silence

def get_no_silence_index_from_one_hot(one_hot_labels_list, type):
    raw_labels_list = [one_hot_to_label(label, type) for label in one_hot_labels_list]
    index_no_silence = [index for index, element in enumerate(raw_labels_list) if element != "silence"]
    return index_no_silence

############ HOW TO USE

# # Exemple de jeu de données
# test_dialogAtc = ["silence", "Declaration", "Backchannel", "Agree/accept" , "Disagree/disaccept", "Question", "Directive" , "Non-understanding", 
#               "Opening", "Apology", "Thanking"]

# # Convertir les labels en one-hot
# one_hot_labels = [label_to_one_hot(label, label_to_index_dialog_act) for label in test_dialogAtc]

# # Convertir les données en tenseurs PyTorch
# label_tensor = torch.stack(one_hot_labels)

# # Utiliser les données et les labels encodés
# print("Exemples de données avec les labels one-hot correspondants (tenseurs PyTorch) :")
# for i in range(len(label_tensor)):
#     example_label = label_tensor[i]
#     print("Label one-hot:", example_label)

# # Utiliser les représentations one-hot pour récupérer les labels
# print("\nRécupération des labels à partir des représentations one-hot :")
# for one_hot_label in label_tensor:
#     recovered_label = one_hot_to_label(one_hot_label, index_to_label_dialog_act)
#     print("Label récupéré à partir de l'encodage one-hot :", recovered_label)
