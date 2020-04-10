import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from tqdm import tqdm

EVAL_METRIC_DICT = {
    'interpretability': 'Interpretability',
    'modularity_score': 'Modularity',
    'mig': 'MIG',
    'SAP_score': 'SAP',
    'Corr_score': 'SCC',
}


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information.
    Args:
        mus: np.array num_points x num_points
        ys: np.array num_points x num_attributes
    """
    num_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    m = np.zeros([num_codes, num_attributes])
    for i in range(num_codes):
        for j in range(num_attributes):
            m[i, j] = mutual_info_score(ys[:, j], mus[:, i])
    return m


def continuous_mutual_info(mus, ys):
    """Compute continuous mutual information.
    Args:
        mus: np.array num_points x num_points
        ys: np.array num_points x num_attributes
    """
    num_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    m = np.zeros([num_codes, num_attributes])
    for i in tqdm(range(num_attributes)):
        m[:, i] = mutual_info_regression(mus, ys[:, i])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information.
    Args:
        ys: np.array num_points x num_attributes
    """
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(ys[:, j], ys[:, j])
    return h


def continuous_entropy(ys):
    """Compute continuous mutual entropy
    Args:
        ys: np.array num_points x num_attributes
    """
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    for j in tqdm(range(num_factors)):
        h[j] = mutual_info_regression(
            ys[:, j].reshape(-1, 1), ys[:, j]
        )
    return h


def compute_interpretability_metric(latent_codes, attributes, attr_list):
    """
    Computes the interpretability metric for each attribute
    Args:
        latent_codes: np.array num_points x num_codes
        attributes: np.array num_points x num_attributes
        attr_list: list of string corresponding to attribute names
    """
    interpretability_metrics = {}
    total = 0
    for i, attr_name in tqdm(enumerate(attr_list)):
        attr_labels = attributes[:, i]
        mutual_info = mutual_info_regression(latent_codes, attr_labels)
        dim = np.argmax(mutual_info)

        # compute linear regression score
        reg = LinearRegression().fit(latent_codes[:, dim:dim + 1], attr_labels)
        score = reg.score(latent_codes[:, dim:dim + 1], attr_labels)
        interpretability_metrics[attr_name] = (int(dim), float(score))
        total += float(score)
    interpretability_metrics["mean"] = (-1, total / len(attr_list))
    return interpretability_metrics


def compute_mig(latent_codes, attributes):
    """
    Computes the mutual information gap (MIG) metric
    Args:
        latent_codes: np.array num_points x num_codes
        attributes: np.array num_points x num_attributes
    """
    score_dict = {}
    m = continuous_mutual_info(latent_codes, attributes)
    entropy = continuous_entropy(attributes)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["mig"] = np.mean(
        np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:])
    )
    return score_dict


def compute_modularity(latent_codes, attributes):
    """
    Computes the modularity metric
    Args:
        latent_codes: np.array num_points x num_codes
        attributes: np.array num_points x num_attributes
    """
    scores = {}
    mi = continuous_mutual_info(latent_codes, attributes)
    scores["modularity_score"] = _modularity(mi)
    return scores


def _modularity(mutual_information):
    """
    Computes the modularity from mutual information.
    Args:
        mutual_information: np.array num_codes x num_attributes
    """
    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.)
    delta = numerator / denominator
    modularity_score = 1. - delta
    index = (max_squared_mi == 0.)
    modularity_score[index] = 0.
    return np.mean(modularity_score)


def compute_correlation_score(latent_codes, attributes):
    """
    Computes the correlation score
    Args:
        latent_codes: np.array num_points x num_codes
        attributes: np.array num_points x num_attributes
    """
    corr_matrix = _compute_correlation_matrix(latent_codes, attributes)
    scores = {
        "Corr_score": np.mean(np.max(corr_matrix, axis=0))
    }
    return scores


def _compute_correlation_matrix(mus, ys):
    """
    Compute correlation matrix for correlation score metric
    """
    num_latent_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    score_matrix = np.zeros([num_latent_codes, num_attributes])
    for i in tqdm(range(num_latent_codes)):
        for j in range(num_attributes):
            mu_i = mus[:, i]
            y_j = ys[:, j]
            rho, p = spearmanr(mu_i, y_j)
            if p <= 0.05:
                score_matrix[i, j] = np.abs(rho)
            else:
                score_matrix[i, j] = 0.
    return score_matrix


def compute_sap_score(latent_codes, attributes):
    """
    Computes the separated attribute predictability (SAP) score
    Args:
        latent_codes: np.array num_points x num_codes
        attributes: np.array num_points x num_attributes
    """
    score_matrix = _compute_score_matrix(latent_codes, attributes)
    # Score matrix should have shape [num_codes, num_attributes].
    assert score_matrix.shape[0] == latent_codes.shape[1]
    assert score_matrix.shape[1] == attributes.shape[1]

    scores = {
        "SAP_score": _compute_avg_diff_top_two(score_matrix)
    }
    return scores


def _compute_score_matrix(mus, ys):
    """
    Compute score matrix for sap score computation.
    """
    num_latent_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    score_matrix = np.zeros([num_latent_codes, num_attributes])
    for i in tqdm(range(num_latent_codes)):
        for j in range(num_attributes):
            mu_i = mus[:, i]
            y_j = ys[:, j]
            # Attributes are considered continuous.
            cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
            cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
            var_mu = cov_mu_i_y_j[0, 0]
            var_y = cov_mu_i_y_j[1, 1]
            if var_mu > 1e-12:
                score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
            else:
                score_matrix[i, j] = 0.
    return score_matrix


def _compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


def normalize_data(data, mean=None, stddev=None):
    """
    Normalizes the data using a z-score normalization
    Args:
        data: np.array
        mean:
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if stddev is None:
        stddev = np.std(data, axis=0)
    return (data - mean[np.newaxis, :]) / stddev[np.newaxis, :], mean, stddev
