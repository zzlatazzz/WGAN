from matplotlib import pyplot as plt
import numpy as np
import sklearn.cluster

from PIL import Image
import io

from sklearn.metrics import auc

def compute_prd(eval_dist, ref_dist, num_angles=1001, epsilon=1e-10):

    if not (epsilon > 0 and epsilon < 0.1):
        raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
    if not (num_angles >= 3 and num_angles <= 1e6):
        raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

    angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
    slopes = np.tan(angles)

    slopes_2d = np.expand_dims(slopes, 1)

    ref_dist_2d = np.expand_dims(ref_dist, 0)
    eval_dist_2d = np.expand_dims(eval_dist, 0)

    precision = np.minimum(ref_dist_2d*slopes_2d, eval_dist_2d).sum(axis=1)
    recall = precision / slopes

    max_val = max(np.max(precision), np.max(recall))
    if max_val > 1.001:
        raise ValueError('Detected value > 1.001, this should not happen.')
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)

    return precision, recall


def _cluster_into_bins(eval_data, ref_data, num_clusters):

    cluster_data = np.vstack([eval_data, ref_data])
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit(cluster_data).labels_

    eval_labels = labels[:len(eval_data)]
    ref_labels = labels[len(eval_data):]

    eval_bins = np.histogram(eval_labels, bins=num_clusters,
                           range=[0, num_clusters], density=True)[0]
    ref_bins = np.histogram(ref_labels, bins=num_clusters,
                          range=[0, num_clusters], density=True)[0]
    return eval_bins, ref_bins


def compute_prd_from_embedding(eval_data, ref_data, num_clusters=20,
                               num_angles=1001, num_runs=10,
                               enforce_balance=True):

    if enforce_balance and len(eval_data) != len(ref_data):
        raise ValueError(
        'The number of points in eval_data %d is not equal to the number of '
        'points in ref_data %d. To disable this exception, set enforce_balance '
        'to False (not recommended).' % (len(eval_data), len(ref_data)))

    eval_data = np.array(eval_data, dtype=np.float64)
    ref_data = np.array(ref_data, dtype=np.float64)
    precisions = []
    recalls = []
    for _ in range(num_runs):
        eval_dist, ref_dist = _cluster_into_bins(eval_data, ref_data, num_clusters)
        precision, recall = compute_prd(eval_dist, ref_dist, num_angles)
        precisions.append(precision)
        recalls.append(recall)
    precision = np.mean(precisions, axis=0)
    recall = np.mean(recalls, axis=0)
    return precision, recall


def compute_auc_pr(precision, recall):
    return auc(np.around(recall, 9), np.around(precision, 9))


def plot(precision_recall_pairs, labels=None, out_path=None, title=None,
         legend_loc='lower left', dpi=300):
 
    if labels is not None and len(labels) != len(precision_recall_pairs):
        raise ValueError(
        'Length of labels %d must be identical to length of '
        'precision_recall_pairs %d.'
        % (len(labels), len(precision_recall_pairs)))

    fig = plt.figure(figsize=(3.5, 3.5), dpi=dpi)
    plot_handle = fig.add_subplot(111)
    plot_handle.tick_params(axis='both', which='major', labelsize=12)

    for i in range(len(precision_recall_pairs)):
        precision, recall = precision_recall_pairs[i]
        label = labels[i] if labels is not None else None
        plt.plot(recall, precision, label=label, alpha=0.5, linewidth=3)

    if labels is not None:
        plt.legend(loc=legend_loc)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, dpi=dpi, bbox_inches='tight', pad_inches=1)
    im = Image.open(img_buf)
 
    if out_path is None:
        plt.close()
    else:
        plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
        plt.close()

    return im