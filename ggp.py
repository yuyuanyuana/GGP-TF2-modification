import numpy as np
import gpflow
import tensorflow as tf
from gpflow.mean_functions import Constant
from scipy.cluster.vq import kmeans2

from GraphSVGP import GraphSVGP
from graph_kernel import GraphPolynomial, NodeInducingPoints
from utils import load_dataset


def training_step(X_train, y_train, q_diag, optimizer, gprocess):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(gprocess.trainable_variables)
        objective = -gprocess.log_likelihood((X_train, y_train), q_diag)
        gradients = tape.gradient(objective, gprocess.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gprocess.trainable_variables))
    return objective


def evaluate(X_val, y_val, q_diag, gprocess):
    pred_y, pred_y_var = gprocess._predict_f_graph(X_val, q_diag)
    pred_classes = np.argmax(pred_y.numpy(), axis=-1)
    # print(y_val)
    y_val = np.argmax(y_val, axis=-1)
    acc = np.mean(pred_classes == y_val)
    return acc


def run_training():
    (adj_matrix, node_feats, node_labels, idx_train, idx_val,
     idx_test) = load_dataset("cora", tfidf_transform=True,
                              float_type=np.float64)

    idx_train = tf.constant(idx_train)
    idx_val = tf.constant(idx_val)
    idx_test = tf.constant(idx_test)
    num_classes = len(np.unique(node_labels.argmax(axis=1)))

    # Init kernel
    kernel = GraphPolynomial(adj_matrix, node_feats, idx_train)

    # Init inducing points
    inducing_points = kmeans2(node_feats, len(idx_train), minit='points')[0]    # use as many inducing points as training samples
    inducing_points = NodeInducingPoints(inducing_points)

    # Init GP model
    mean_function = Constant()
    q_diag = True
    gprocess = GraphSVGP( kernel = kernel, 
                          likelihood = gpflow.likelihoods.Gaussian(),
                          inducing_variable = inducing_points,
                          mean_function=mean_function,
                          num_latent_gps=num_classes, 
                          whiten=True, 
                          q_diag=q_diag)

    # Init optimizer
    optimizer = tf.optimizers.Adam()

    for epoch in range(2000):
        elbo = -training_step(X_train = idx_train, 
                              y_train = node_labels[idx_train], 
                              q_diag = q_diag, 
                              optimizer = optimizer,
                              gprocess = gprocess)
        elbo = elbo.numpy()

        acc = evaluate(idx_test, node_labels[idx_test], q_diag, gprocess)
        print(f"{epoch}:\tELBO: {elbo:.5f}\tAcc: {acc:.3f}")


if __name__ == '__main__':
    run_training()