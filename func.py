import random
import re 
import sympy
import math
import numpy as np 
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
from clustering import cluster_method


def run_iteration(tv_list, tv_list_test, t, number_of_hashes):
    # set brands
    set_brands(tv_list)

    # extract model words 
    model_words = title_models_words(tv_list)

    # extract all brands
    brands = extract_all_brands(tv_list)

    # extract feature model words
    feature_words = extract_model_words_features(tv_list)

    # create binary matrix
    binary = binary_matrix(model_words, feature_words, brands, tv_list)
    
    # create signature matrix
    signature = signature_matrix(binary, number_of_hashes)

    # find optimal row value 
    r = find_r(len(signature[0]), t)

    # candidates 
    candidates = lsh_method(signature, r, tv_list)

    alpha = 0.6
    gamma_options = [0.75]
    mu_options = [0.5, 0.6, 0.7]
    epsilons = [0.4, 0.5, 0.6]

    f1_final = 0
    mu_hyperparameter = 0
    epsilon_hyperparameter = 0

    for gamma in gamma_options:
        for mu in mu_options:

            # run msm
            distance_matrix_clustering = cluster_method(tv_list, candidates, gamma, alpha, mu)

            highest_f1 = 0
            mu_opt = 0
            epsilon_opt = 0

            for epsilon in epsilons:
                clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=epsilon, 
                                    metric='precomputed', linkage='single').fit(distance_matrix_clustering)
                labels = clustering.labels_
                clusters = get_clusters(labels)
        

                # msm performance
                f1, _, _ = performance_MSMP(clusters, tv_list)
                if f1 > highest_f1:
                    highest_f1 = f1
                    mu_opt = mu
                    epsilon_opt = epsilon
            
            if highest_f1 > f1_final:
                f1_final = highest_f1
                mu_hyperparameter = mu_opt
                epsilon_hyperparameter = epsilon_opt
    
    result = run_test_set(tv_list_test, t, number_of_hashes, alpha, gamma, mu_hyperparameter, epsilon_hyperparameter)
    return result


def performance_MSMP(clusters, tvs):
    true_pairs = get_true_duplicates(tvs)
    TP = 0
    FP = 0
    FN = 0

    predicted_pairs = []
    for cluster in clusters.values():
        if len(cluster) > 1:
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    pair = (cluster[i], cluster[j])
                    predicted_pairs.append(pair)

    for pair in predicted_pairs:
        i, j = pair 
        if tvs[i] == tvs[j]:
            TP += 1
        else:
            FP += 1
    
    for true_dup in true_pairs:
        if true_dup not in predicted_pairs:
            FN += 1

    if TP > 0 or FP > 0:
        f1 = (2 * TP) / (2 * TP + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    else:
        f1 = precision = recall = 0
    return f1, precision, recall


def lsh_performance(candidates, tv_list):
    true_duplicates = get_true_duplicates(tv_list)

    TP = 0
    FP = 0
    for candidate in candidates:
        i, j = candidate
        if i != j: 
            tv1 = tv_list[i]
            tv2 = tv_list[j]
            if tv1 == tv2: 
                TP += 1
        else:
            FP += 1
    FN = 0    
    for true_dup in true_duplicates:
        i,j = true_dup
        if [i,j] not in candidates:
            FN += 1

    PQ = TP / len(candidates)
    PC = TP / len(true_duplicates)
    fraction_comparisons = len(candidates) / math.comb(len(tv_list), 2)
    if PC > 0 or PQ > 0:
        f1star = 2 * PQ * PC / (PC + PQ)
    else:
        f1star = 0
    return PQ, PC, f1star, fraction_comparisons


def lsh_method(matrix, r, all_tvs):
    bands = split_matrix(matrix, r)

    buckets = {}
    for band in bands:
        number_of_values = len(band[0])
        for p, _ in enumerate(all_tvs):
            signature = ""
            for hash_number in range(number_of_values):
                signature += str(band[p][hash_number])

            existing_list = buckets.get(signature)
            if existing_list is not None:
                existing_list.append(p)
            else:
                buckets[signature] = [p]
    

    candidate_pairs = []

    for _, values in buckets.items():
        if len(values) > 1:  # Only consider pairs if the list has more than one element
            for pair in combinations(values, 2):
                i,j = pair
                if i != j:
                    candidate_pairs.append(sorted(pair))

    # Remove duplicate pairs
    candidate_pairs = list(set(map(tuple, candidate_pairs)))
    candidate_pairs = [list(pair) for pair in candidate_pairs]
    return candidate_pairs


def signature_matrix(input_matrix, number_of_hashes):
    N = len(input_matrix[0])
    P = len(input_matrix)
    matrix = np.full((P, number_of_hashes), np.inf)
    prime = sympy.nextprime(N * 5)

    np.random.seed(10)
    hash_funcs = [lambda x, a=a, b=b: (a * x + b) % prime for a, b in zip(np.random.randint(1, prime, number_of_hashes), np.random.randint(0, prime, number_of_hashes))]

    for column in range(N):
        row_number_hash = column + 1
        hash_values = []
        for h in range(number_of_hashes):
            hash_value = hash_funcs[h](row_number_hash)
            hash_values.append(hash_value)
        for row in range(P):
            if input_matrix[row][column] == 1:
                for index, value in enumerate(hash_values):
                    if value < matrix[row][index]:
                        matrix[row][index] = value

    matrix = matrix.astype(int)
    return matrix


def binary_matrix(model_words_title, model_words_features, brands_list, tv_list): 
    # matrix has rows of length models_words and feature shingles, columns of length tv_list products
    P = len(tv_list)
    N1 = len(model_words_title)
    N2 = len(model_words_features)
    N3 = len(brands_list)
    
    matrix = [] 
    for j in range(P):
        vector_rep = np.zeros(N1 + N2 + N3)
        for i1 in range(N1): 
            if model_words_title[i1] in tv_list[j].get_title().lower():
                vector_rep[i1] = 1

        for i2 in range(N2):
            vector_rep_index = i2 + N1
            if 'Screen Size Class' in tv_list[j].get_features().keys():
                if model_words_features[i2]  == tv_list[j].get_features()['Screen Size Class']:
                    vector_rep[vector_rep_index] = 1
            if 'ENERGY STAR Certified' in tv_list[j].get_features().keys():
                if model_words_features[i2]  == tv_list[j].get_features()['ENERGY STAR Certified']:
                    vector_rep[vector_rep_index] = 1

        for i3 in range(N3):
            vector_rep_index = N1 + N2 + i3
            if brands_list[i3] in tv_list[j].get_title().lower():
                vector_rep[vector_rep_index] = 1
            if 'Brand' in tv_list[j].get_features().keys():
                if brands_list[i3] == tv_list[j].get_features()['Brand']:
                    vector_rep[vector_rep_index] = 1
            if 'Brand Name' in tv_list[j].get_features().keys():
                if brands_list[i3] == tv_list[j].get_features()['Brand Name']:
                    vector_rep[vector_rep_index] = 1            
        matrix.append(vector_rep) 
    matrix = np.array(matrix)
    return matrix


def extract_model_words_features(tv_list):
    model_words = []
    for tv in tv_list:
        for key, value in tv.get_features().items():
            if key == 'Screen Size Class' or key == 'ENERGY STAR Certified' or key == 'Width':
                word = re.findall(r'\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]+\b', value)
                if word not in model_words and len(word) != 0:
                    model_words.append(word[0])
    return list(set(model_words))


def title_models_words(tv_list):
    model_words_list = []
    for tv in tv_list: 
        model_words = re.findall(r'\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]+\b', tv.get_title())
        for word in model_words:
            if word.lower() not in model_words_list:
                model_words_list.append(word.lower())
    return model_words_list


def extract_all_brands(tvs):
    brand_list = []
    for tv in tvs:
        brand = tv.get_features().get("Brand") or tv.get_features().get("Brand Name")
        if brand is not None and brand.lower() not in brand_list:
            brand_list.append(brand.lower())
    return brand_list
    

def set_brands(tvs):
    brand_to_set = None
    brands = extract_all_brands(tvs)
    for tv in tvs:
        for brand_name in brands: 
            if brand_name.lower() in tv.title.lower():
                brand_to_set = brand_name.lower()
        
        if tv.features.get("Brand"):
            brand_to_set = tv.features.get("Brand")
        elif tv.features.get("Brand Bame"):
            brand_to_set = tv.features.get("Brand Name")
        
        if brand_to_set is not None:
            tv.brandname = brand_to_set.lower()
        else: 
            tv.brandname = " - "


def bootstrap(products):
    bootstrap_length = len(products)
    
    train_set = []
    test_set = []
    index_list = set()

    for _ in range(bootstrap_length):
        random_index = random.randint(0, bootstrap_length - 1)
        
        if random_index not in index_list:
            train_set.append(products[random_index])
            index_list.add(random_index)

    test_set = [product for i, product in enumerate(products) if i not in index_list]
    return train_set, test_set


def find_r(N, t):
    best_r = 1
    best_distance = 1
    
    for rows in range(1, N + 1):
        band = N // rows  # Integer division to get a whole number for b
        if rows * band == N:
            t_approx = math.pow((1/band), (1/rows))
            distance = abs(t - t_approx)
            if distance < best_distance:
                best_r = rows
                best_distance = distance

    return best_r


def split_matrix(matrix, b):
    result_matrices = np.array_split(matrix, np.arange(b, matrix.shape[1], b), axis=1)
    return result_matrices


def get_true_duplicates(tv_list):
    true_dup = []
    for i, tv1 in enumerate(tv_list):
        for j, tv2 in enumerate(tv_list):
            if tv1 == tv2 and i < j:
                true_dup.append((i, j))
    return true_dup


def get_clusters(labels):
    cluster_dict = {}
    for index, cluster in enumerate(labels):
        if cluster in cluster_dict:
            cluster_dict[cluster].append(index)
        else:
            cluster_dict[cluster] = [index]
    return cluster_dict



def run_test_set(tv_list, t, number_of_hashes, alpha, gamma, mu, epsilon):
    # set brands
    set_brands(tv_list)

    # extract model words 
    model_words = title_models_words(tv_list)

    # extract all brands
    brands = extract_all_brands(tv_list)

    # extract feature model words
    feature_words = extract_model_words_features(tv_list)

    # create binary matrix
    binary = binary_matrix(model_words, feature_words, brands, tv_list)
    
    # create signature matrix
    signature = signature_matrix(binary, number_of_hashes)

    # find optimal row value 
    r = find_r(len(signature[0]), t)

    # candidates 
    candidates = lsh_method(signature, r, tv_list)

    # lsh performance 
    PQ, PC, f1star, fraction_comparisons = lsh_performance(candidates, tv_list)

    # distance matrix 
    distance_matrix_clustering = cluster_method(tv_list, candidates, gamma, alpha, mu)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=epsilon, 
                                         metric='precomputed', linkage='single').fit(distance_matrix_clustering)
    labels = clustering.labels_
    clusters = get_clusters(labels)
    f1, precision, recall = performance_MSMP(clusters, tv_list)

    return [f1, precision, recall, f1star, PC, PQ, fraction_comparisons]