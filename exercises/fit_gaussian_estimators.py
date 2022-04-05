from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

NUM_SAMPLES = 1000
# Constants for Univariate gaussian
UNI_MU, UNI_SIGMA = 10, 1
NEW_SAMPLES_AMOUNT, INIT_VAL, MAX_VAL, NUM_OF_FITS = 10, 1, 101, 100

# Constants for Multivariate gaussian
MULTI_MU = np.array([0, 0, 4, 0])
MULTI_SIGMA = np.matrix([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni_samples = np.random.normal(UNI_MU, UNI_SIGMA, NUM_SAMPLES)
    univariate_q1 = UnivariateGaussian()
    univariate_q1.fit(uni_samples)
    print("(" + str(univariate_q1.mu_) + ", " + str(univariate_q1.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    univariate_q2 = UnivariateGaussian()
    num_samples = np.linspace(NEW_SAMPLES_AMOUNT, NUM_SAMPLES, NUM_OF_FITS)
    absolute_distance_arr = []
    # calculate the models
    for sample_num in range(INIT_VAL, MAX_VAL):
        current_samples = uni_samples[: NEW_SAMPLES_AMOUNT * sample_num]
        univariate_q2.fit(current_samples)
        absolute_distance_arr.append(np.abs(UNI_MU - univariate_q2.mu_))

    # plot the graph
    go.Figure(go.Scatter(x=num_samples, y=absolute_distance_arr, mode='markers+lines'),
              layout=go.Layout(
                  title=r"$\text{Absolute Distance As A Function Of The Sample Size}$",
                  xaxis_title="$\\text{Number Of Samples}$",
                  yaxis_title="$\\text{Absolute Distance}$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    univariate_pdf = univariate_q1.pdf(uni_samples)
    # plot the graph
    go.Figure(go.Scatter(x=uni_samples, y=univariate_pdf, mode='markers'),
              layout=go.Layout(
                  title=r"$\text{Empirical PDF Function Under the Fitted Model}$",
                  xaxis_title="$\\text{Samples Value}$",
                  yaxis_title="$\\text{PDF}$",
                  height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multi_samples = np.random.multivariate_normal(MULTI_MU, MULTI_SIGMA, NUM_SAMPLES)
    multi_variate = MultivariateGaussian()
    multi_variate.fit(multi_samples)
    print(multi_variate.mu_)
    print(multi_variate.cov_)

    # # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    n = f1.size
    log_likelihood = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mu = np.array([f1[i], 0, f3[j], 0])
            log_likelihood[i, j] = multi_variate.log_likelihood(mu, MULTI_SIGMA, multi_samples)
    # Plot the heatmap
    go.Figure(data=go.Heatmap(
        z=log_likelihood, x=f1, y=f3),
        layout=go.Layout(
            title=r"$\text{Log Likelihood Heatmap Using F1 and F3}$",
            xaxis_title="$\\text{F1}$",
            yaxis_title="$\\text{F3}$",
            height=600, width=600)).show()

    # Question 6 - Maximum likelihood
    max_arg_likelihood = np.argmax(log_likelihood)
    index_max_arg = np.unravel_index(max_arg_likelihood, log_likelihood.shape)
    print("(" + str(f1[index_max_arg[0]]) + ", " + str(f3[index_max_arg[1]]) + ")")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
