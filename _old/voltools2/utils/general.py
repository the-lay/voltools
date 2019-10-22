import string
import random
import sys


def second_to_h_m_s(time: int) -> (int, int, int):
    # https://github.com/pytorch/ignite/blob/master/ignite/_utils.py
    mins, secs = divmod(time, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs

def generate_random_id() -> str:
    # https://stackoverflow.com/questions/13484726/safe-enough-8-character-short-unique-random-string
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=8))

def generate_random_seed() -> int:
    return random.randint(-sys.maxsize - 1, sys.maxsize)

def readable_size(file_size: int) -> str:
    for unit in ['', 'K', 'M', 'B']:
        if file_size < 1000:
            break
        file_size /= 1000
    return f'{file_size:.3f}{unit}'
