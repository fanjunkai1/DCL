
def read_lines(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]
