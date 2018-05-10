
def evaluate_device(device):
    return "/device:CPU:0" if device == "cpu" else "/device:GPU:0"
