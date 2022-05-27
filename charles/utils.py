from matplotlib import pyplot as plt


def plot_c(c, alpha, threshold):
    """ A function to visualize the changing c value (vertical) and
    number of updates (horizontal)

    Args:
        c: temperature parameter
        alpha: decreasing factor
        threshold: threshold for termination condition
    """
    c_list = [c]
    while c > threshold:
        c = c * alpha
        c_list.append(c)
    plt.plot(c_list)
    plt.show()


plot_c(100, 0.8, 0.05)