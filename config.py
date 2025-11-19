# the sets we are using to test (2,022 cards in total)
SETS: list[str] = ['fut', 'uma', 'rna', 'war', 'ktk', 'm19', 'otj', 'blb']

# bin versions we are testing out
# (we use more bins for the hue since it's more important for color differentiation)
BIN_VERSIONS: list[tuple[str, tuple[int, int, int]]] = [
    ('bin_A', (8, 8, 8)),    # 512 features
    ('bin_B', (16, 16, 16)), # 4096 features
    ('bin_C', (16, 8, 8)),   # 1024 features
    ('bin_D', (32, 8, 8)),   # 2048 features
    ('bin_E', (4, 4, 4)),    # 64 features
    ('bin_F', (8, 4, 4)),    # 128 features
    ('bin_G', (16, 4, 4)),   # 256 features
]
