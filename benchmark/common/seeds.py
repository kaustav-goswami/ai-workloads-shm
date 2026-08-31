def deterministic_seed(name):
    seed = 0
    for char in name:
        seed = (seed * 131 + ord(char)) % (2**31)
    return seed
