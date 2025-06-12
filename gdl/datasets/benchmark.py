# Benchmark dataset configuration for different fields across different scenarios


def get_field_D_grid_split():
    """Get the split indices of the grid cells within field D.
    ToDo: move this function to benchmark.py.

    Returns:
        list: List of grid cell indices.
    """
    test_indices = [54, 50, 39, 37, 25]
    train_indices = [
        3,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        23,
        24,
        26,
        27,
        29,
        30,
        31,
        32,
        33,
        34,
        36,
        38,
        40,
        41,
        43,
        44,
        45,
        46,
        47,
        48,
        51,
        52,
        53,
        55,
        57,
        58,
        59,
        60,
        61,
        62,
    ]
    return train_indices, test_indices
