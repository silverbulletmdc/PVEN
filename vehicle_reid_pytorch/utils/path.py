import os


def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)
