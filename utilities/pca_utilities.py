from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def get_pca_data(X, X_test):
    pca = PCA(90)
    principal_components =pca.fit_transform(X)
    principal_components_test =pca.transform(X_test)
    print(f'Kept: {principal_components.shape[1]} components')

    visualize_2D_components(principal_components)
    plot_explained_variance(pca)
    return principal_components, principal_components_test

def plot_explained_variance(pca):

    plt.plot(pca.explained_variance_ratio_)
    plt.savefig("explained_variance.png")

def visualize_2D_components(principal_components):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    ax.scatter(principal_components[0] ,principal_components[1], c='r' , s=50)

    ax.grid()
    plt.savefig("pca.png")