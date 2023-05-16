import sys
import scipy
import matplotlib.pyplot as plt


#
# Usage: python visualize_simtb.py <FILENAME>.mat
#

if __name__ == '__main__':
    data_path = sys.argv[1]
    data = scipy.io.loadmat(data_path)
    source_maps = data['SM']

    reshaped_source_maps = source_maps.reshape(20, 128, 128)
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    axes = axes.flatten()
    for i in range(20):
        axes[i].imshow(reshaped_source_maps[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(str(i + 1), fontsize=10, pad=2)
    plt.tight_layout()
    plt.show()
