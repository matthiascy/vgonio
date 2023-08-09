import numpy as np


class BeckmannSpizzichinoModel:
    def __init__(self, alpha, n):
        self.alpha = alpha
        self.normal = n

    def eval_ndf_with_m(self, m):
        return self.eval_ndf_with_cos_theta_m(m.dot(self.normal))

    def eval_ndf_with_theta_m(self, theta_m):
        return self.eval_ndf_with_cos_theta_m(np.cos(theta_m))

    def eval_ndf_with_cos_theta_m(self, cos_theta_m):
        alpha2 = self.alpha * self.alpha
        cos_theta_m2 = cos_theta_m * cos_theta_m
        tan_theta_m2 = (1.0 - cos_theta_m2) / cos_theta_m2
        cos_theta_m4 = cos_theta_m2 * cos_theta_m2
        return np.exp(-tan_theta_m2 / alpha2) / (np.pi * alpha2 * cos_theta_m4)


if __name__ == "__main__":
    alpha = 0.5872
    model = BeckmannSpizzichinoModel(alpha, np.array([0.0, 1.0, 0.0]))
    thetas = np.linspace(-np.pi/2, np.pi/2, 100)
    evaluate = np.vectorize(lambda theta: model.eval_ndf_with_theta_m(theta))
    line = evaluate(thetas)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()
    plt.title(f"BeckmannSpizzichino NDF: α = {alpha}")
    plt.plot(thetas, line)
    plt.show()