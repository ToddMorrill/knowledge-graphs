import hashlib

BUF_SIZE = 65536  # 2^16 bytes, read file in chunks for hexdigest


def hms_string(sec_elapsed: int) -> str:
    """Nicely formatted time string.

    Args:
        sec_elapsed (int): Integer number of seconds.

    Returns:
        str: h:m:s formatted string.
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def strip_tag_name(t):
    idx = k = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t


def sha1_file(path):
    sha1 = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()