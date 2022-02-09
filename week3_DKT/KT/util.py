import csv
import glob
import os

from sklearn.utils import shuffle
import numpy as np
import random


def create_full_path(user_base_path, user_path):
    u0 = user_path[0]
    u1 = user_path[1]
    u2 = user_path[2]
    u3 = user_path[3]
    return f'{user_base_path}/{u0}/{u1}/{u2}/{u3}/{user_path}'


def get_qid_to_embed_id(dict_path):
    d = {}
    with open(dict_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split(',')
            d[int(line[0])] = int(line[1])
    return d


def get_sample_info(user_base_path, data_path):
    # for modified_AAAI20 data
    sample_infos = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        num_of_users = len(lines)
        for user_path in lines:
            user_path = user_path.rstrip()
            user_full_path = create_full_path(user_base_path, user_path)
            with open(user_full_path, 'r', encoding='ISO-8859-1') as f:
                lines = f.readlines()
                num_of_interactions = len(lines)
                for target_index in range(num_of_interactions):
                    sample_infos.append([user_path, target_index])

    return sample_infos, num_of_users

# Do not use this anymore
def get_data_tl(data_path):
    # for triple line format data
    data_list = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        num_of_users = len(lines) // 3
        for i in range(num_of_users):
            user_interaction_len = int(lines[3*i].strip())
            qid_list = list(map(int, lines[3*i+1].split(',')))
            is_correct_list = list(map(int, lines[3*i+2].split(',')))
            assert user_interaction_len == len(qid_list) == len(is_correct_list), 'length is different'

            for j in range(user_interaction_len):
                data_list.append((qid_list[:j+1], is_correct_list[:j+1]))

    return data_list, num_of_users


def get_data_user_sep(data_path, ratio):
    # almost same as get_sample_info
    # for user separated format data
    sample_infos = []
    # get list of all files
    user_path_list = os.listdir(data_path)
    num_of_users = len(user_path_list)
    print("prev", num_of_users)
    
    # use part of data
    num_of_users = int(num_of_users * ratio)
    random.seed()
    user_path_list = random.sample(user_path_list, num_of_users)
    print("new", num_of_users)
    
    total_false = 0
    for user_path in user_path_list:
        with open(data_path + user_path, 'rb') as f:
            lines = f.readlines()
            lines = lines[1:]
            # print(lines)
            num_of_interactions = len(lines)

            for end_index in range(num_of_interactions):
                answer = 1
                if(lines[end_index].decode()[lines[end_index].find(','.encode())+1] == "0"):
                    total_false += 1
                    answer = 0

                sample_infos.append((data_path + user_path, end_index, answer))

    random.shuffle(sample_infos)
    a = np.array(sample_infos)
    new_a = np.delete(a,(np.array(np.where(a[:,2] == '1')).reshape(-1)[:len(sample_infos)-2*total_false]), axis=0)
    new_a = np.delete(new_a, 2, 1)
    sample_infos = new_a.tolist()
    for row in sample_infos:
        row[1] = int(row[1])

    # random.shuffle(sample_infos)
    # delete_total_real = len(sample_infos)-2*total_false
    # delete_total = 0
    # for index, tuple in enumerate(sample_infos):
    #     # print(tuple[2])
    #     if tuple[2] == 1:
    #         delete_total += 1
    #         del sample_infos[index]
        
    #     # print(delete_total_real, total_false, delete_total)
    #     if(delete_total_real == total_false):
    #         break
    # print(sample_infos)    

    # correct_total = 0
    # incorrect_total = 0
    # for index, tuple in enumerate(sample_infos):
    #     if tuple[2] == '1':
    #         correct_total += 1
    #     else:
    #         incorrect_total += 1
    # print("correct", correct_total, "false", incorrect_total)

    # print("correct", len(sample_infos)-total_false, "false", total_false)
    print(len(sample_infos))

    return sample_infos, num_of_users
