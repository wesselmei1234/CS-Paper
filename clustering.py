import re
import math
import numpy as np 
from Levenshtein import distance


def cluster_method(tvs, candidate_pairs, key_threshold, alpha, mu):
    matrix_size = len(tvs)
    dist_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        dist_matrix[i][i] = 10000000000

    for i, tv1 in enumerate(tvs):
        for j, tv2 in enumerate(tvs[i+1:]):
            j += i + 1

            if same_shop(tv1, tv2):
                dist_matrix[i, j] = 10000000000
                dist_matrix[j, i] = 10000000000
            elif diff_brand(tv1, tv2):
                dist_matrix[i, j] = 10000000000
                dist_matrix[j, i] = 10000000000
            elif diff_screen_size_class(tv1, tv2):
                dist_matrix[i, j] = 10000000000
                dist_matrix[j, i] = 10000000000
            elif diff_max_res(tv1, tv2):
                dist_matrix[i, j] = 10000000000
                dist_matrix[j, i] = 10000000000
            elif [i, j] not in candidate_pairs:
                dist_matrix[i, j] = 10000000000
                dist_matrix[j, i] = 10000000000
            elif find_possible_modelid(tv1, tv2):
                dist_matrix[i, j] = 0.001
                dist_matrix[j, i] = 0.001
            else:
                sim = 0
                avg_sim = 0
                m = 0
                w = 0 
                features_product_i = tv1.get_features()       
                features_product_j = tv2.get_features()

                for key_i, value_i in features_product_i.items():
                    for key_j, value_j in features_product_j.items(): 
                        key_similarity = calc_sim(key_i, key_j)
                        if key_similarity > key_threshold:
                            value_similarity = calc_sim(value_i, value_j)
                            weight = key_similarity
                            sim += weight * value_similarity
                            m += 1
                            w += weight
                
                if w > 0:
                    avg_sim = sim / w

                title_sim = find_title_sim(tv1, tv2, alpha)

                theta_1 = 1 - mu
                h_sim = theta_1 * avg_sim + mu * title_sim 
                    
                dist = 1 - h_sim
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist

    return dist_matrix


def find_possible_modelid(tv1, tv2):
    title_mw_tv1 = list(set(extract_model_words_title(tv1.get_title())))
    title_mw_tv2 = list(set(extract_model_words_title(tv2.get_title())))
    longest_mw1 = max(title_mw_tv1, key=len)
    longest_mw2 = max(title_mw_tv2, key=len)
    if any(char.islower() for char in longest_mw1) or any(char.islower() for char in longest_mw2):
        return False
    if longest_mw1 == longest_mw2:
        return True
    return False


def extract_model_words_title(title):
    model_words = re.findall(r'\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]+\b', title)
    return model_words


def find_title_sim(tv1, tv2, alpha): 
    title_tv1 = tv1.get_title().split()
    title_tv2 = tv2.get_title().split()
    length_title1 = len(title_tv1)
    length_title2 = len(title_tv2)
    length_intersection = len(set(title_tv1).intersection(title_tv2))

    cosine_sim = length_intersection / (math.sqrt(length_title1) * math.sqrt(length_title2))
    if cosine_sim > alpha:
        return 1  
        
    title_mw_tv1 = list(set(extract_model_words_title(tv1.get_title())))
    title_mw_tv2 = list(set(extract_model_words_title(tv2.get_title())))
        
    model_word_pairs = pairs_nested(title_mw_tv1, title_mw_tv2)

    avg_lev_sim = get_avg_lev_sim(model_word_pairs)
    return avg_lev_sim

def pairs_nested(mw1, mw2):
    pairs = []
    for item1 in mw1:
        for item2 in mw2:
            pairs.append((item1, item2))

    # Using list comprehension
    pairs_comprehension = [(item1, item2) for item1 in mw1 for item2 in mw2]
    return pairs_comprehension

def get_avg_lev_sim(pairs):
    total = 0
    length_of_all_strings = 0
    for pair in pairs:
        title_word_i, title_word_j = pair
        lev_distance_complement =  1 - distance(title_word_i, title_word_j) / max(len(title_word_i), len(title_word_j))
        total += lev_distance_complement * (len(title_word_i) + len(title_word_j))
        length_of_all_strings += (len(title_word_i) + len(title_word_j))
    return (total / length_of_all_strings)


def extract_non_numeric_part(string): 
    non_numeric_part = re.search(r'\D+', string)
    if non_numeric_part != None:
        return non_numeric_part.group()
    else:
        return ""
    

def extract_numeric_part(string): 
    numeric_part = re.search(r'\d+', string)
    return numeric_part.group() 


def mw_perc(model_words_i, model_words_j): 
    set_i = set(model_words_i)
    set_j = set(model_words_j)
    intersection_size = len(set_i.intersection(set_j))
    union_size = len(set_i.union(set_j))

    if intersection_size > 0:
        percentage_common = (intersection_size / union_size)
    else:
        percentage_common = 0
    return percentage_common


def calc_sim(key1, key2):
    k = 3
    shingles_key1 = shingle_set(k, key1.lower().replace(" ", ""))
    shingles_key2 = shingle_set(k, key2.lower().replace(" ", ""))
    n1 = len(shingles_key1)
    n2 = len(shingles_key2)
    qGramOverlap = len(set(shingles_key1) & set(shingles_key2))
    q_gram_distance = n1 + n2 - 2 * qGramOverlap
    similarity = 0
    if n1 + n2 != 0:
        similarity = (n1 + n2 - q_gram_distance) / (n1 + n2)
    return similarity


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


def same_shop(tv1, tv2):
    if tv1.get_shop() == tv2.get_shop():
        return True
    else:
        return False 


def diff_brand(tv1, tv2): 
    if tv1.brandname == None:
        return False
    elif tv1.brandname != tv2.brandname:
        return True 
    else: 
        return False
    

def diff_screen_size_class(tv1, tv2):
    if 'Screen Size Class' in tv1.get_features() and 'Screen Size Class' in tv2.get_features():
        if tv1.get_features()['Screen Size Class'] != tv2.get_features()['Screen Size Class']:
            return True
    return False 

def diff_max_res(tv1, tv2): 
    if 'Maximum Resolution' in tv1.get_features() and 'Maximum Resolution' in tv2.get_features():
        if tv1.get_features()['Maximum Resolution'] != tv2.get_features()['Maximum Resolution']:
            return True
    return False 

def shingle_set(k, value):
    set_of_shingles = []
    for i in range(len(value)):
        if (len(value[i:k + i]) == k):
            set_of_shingles.append(value[i:k + i])
    return list(set(set_of_shingles))