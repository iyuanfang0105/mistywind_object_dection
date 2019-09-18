def get_mscoco_label_map(file_path):
    with open(file_path, 'r', encoding='utf-8') as fin:
        data = fin.readlines()

    label_map = {}
    for d in data:
        id, name = d.strip().split(sep="##")
        label_map[int(id)] = {'id': id, 'name': name}
    return label_map


if __name__ == '__main__':
    file_path = '../images/mscoco_label_map.txt'
    label_map = get_mscoco_label_map(file_path)