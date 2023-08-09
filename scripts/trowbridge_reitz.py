import numpy as np
from lmfit import Parameters, minimize


# from lmfit import minimize, Parameters


class TrowbridgeReitzModel:
    @classmethod
    def eval_adf_with_m(cls, alpha, m, n):
        return cls.eval_adf_with_cos_theta_m(alpha, np.dot(m, n))

    @classmethod
    def eval_adf_with_theta_m(cls, alpha, theta_m):
        return cls.eval_adf_with_cos_theta_m(alpha, np.cos(theta_m))

    @classmethod
    def eval_adf_with_cos_theta_m(cls, alpha, cos_theta_m):
        alpha2 = alpha * alpha
        cos_theta_m2 = cos_theta_m * cos_theta_m
        cos_theta_m4 = cos_theta_m2 * cos_theta_m2
        tan_theta_m2 = (1.0 - cos_theta_m2) / cos_theta_m2
        alpha2_plus_tan_theta_m2 = alpha2 + tan_theta_m2
        return alpha2 / (np.pi * cos_theta_m4 * alpha2_plus_tan_theta_m2 * alpha2_plus_tan_theta_m2)

    @classmethod
    def adf_residual(cls, params, x, data, uncertainty):
        alpha = params['alpha']
        return (data - cls.eval_adf_with_theta_m(alpha, x)) / uncertainty

    @classmethod
    def fit_adf(cls, data, uncertainty):
        params = Parameters()
        params.add("alpha", value=0.1)
        result = minimize(cls.adf_residual, params, args=(data, uncertainty))
        return result.params['alpha'].value


def fit_adf(data):
    params = Parameters()
    params.add("alpha", value=0)
    alpha = 0.5872
    model = TrowbridgeReitzADFModel(alpha, np.array([0.0, 1.0, 0.0]))
    thetas = np.linspace(-np.pi/2, np.pi/2, 100)
    evaluate = np.vectorize(lambda theta: model.eval_adf_with_theta_m(theta))
    line = evaluate(thetas)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()
    plt.title(f"Trowbridge-Reitz NDF: α = {alpha}")
    plt.plot(thetas, line)
    plt.show()


if __name__ == "__main__":
    plot_trowbridge_reitz_ndf()