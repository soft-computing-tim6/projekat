import os
import numpy as np
import json


def split_dataset(test_ratio: float):
    if not(test_ratio > 0.0 and test_ratio < 1.0):
        raise ValueError('Test ratio must be greater than 0 and less than 1')

    annotations_location = os.path.join(os.getcwd(), 'annotations')
    test_file = open(os.path.join(annotations_location, 'test.csv'), 'w')
    validation_file = open(os.path.join(annotations_location, 'validation.csv'), 'w')
    train_file = open(os.path.join(annotations_location, 'train.csv'), 'w')
    data = open(os.path.join(annotations_location, 'data.json')).readlines()

    np.random.seed()
    data_permutations = np.random.permutation(len(data))

    split_length = int(len(data) * test_ratio)
    test_data = data_permutations[:split_length]
    validation_data = data_permutations[split_length:split_length*2]
    train_data = data_permutations[split_length * 2:]

    write_test_to_csv(test_data, data, test_file)
    print('Created test annotation file')

    write_csv_file(validation_data, data, validation_file)
    print('Created validation annotation file')

    write_csv_file(train_data, data, train_file)
    print('Created train annotation file')


def write_csv_file(data, all_data, file):
    for line_number in data:
        line = json.loads(all_data[line_number].strip())
        write_line_to_csv(line, file)
    file.close()


def read_single_person_list():
    f = open(os.path.join(os.getcwd(), "annotations", "single_person_image_ids.txt"), "r")
    image_id_list = f.readlines()
    f.close()
    return image_id_list


def write_line_to_csv(line, file):
    joints = sorted([[int(key), value] for key, value in line['joint_pos'].items()])
    joints = np.array([j for i, j in joints]).flatten()

    csv_data = [line['filename']]
    csv_data.extend(joints)
    
    file.write(','.join([str(field) for field in csv_data]) + '\n')


def write_test_to_csv(data, all_data, file):
    image_ids = read_single_person_list()
    for line_number in data:
        for id in image_ids:
            if id.strip() in all_data[line_number]:
                line = json.loads(all_data[line_number].strip())
                write_line_to_csv(line, file)
    file.close()


if __name__ == "__main__":
    split_dataset(0.1)
