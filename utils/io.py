from utils.EditPath import EditPath


def save_edit_path_to_file(db_name, edit_paths, file_path):
    # save the global edit paths to a file
    with open(f'{file_path}/{db_name}_ged_paths.paths', 'w') as f:
        for i in range(len(edit_paths)):
            for j in range(i + 1, len(edit_paths)):
                if (i, j) not in edit_paths:
                    continue
                else:
                    for edit_path in edit_paths[(i, j)]:
                        f.write(f"{i} {j} {edit_path.toJSON()}\n")

def load_edit_paths_from_file(db_name, file_path):
    # load the global edit paths from a file
    edit_paths = dict()
    with open(f'{file_path}/{db_name}_ged_paths.paths', 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 2)
            if len(parts) < 3:
                continue
            i, j, json_str = int(parts[0]), int(parts[1]), parts[2]
            edit_path = EditPath().loadJSON(eval(json_str))
            if (i, j) not in edit_paths:
                edit_paths[(i, j)] = []
            edit_paths[(i, j)].append(edit_path)
    return edit_paths