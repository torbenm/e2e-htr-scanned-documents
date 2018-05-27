def split(data, dev_frac, test_frac):
    l = len(data)
    dev_len = int(l * dev_frac)
    test_len = int(l * test_frac)
    return data[dev_len + test_len:], data[:dev_len], data[dev_len:dev_len + test_len]
