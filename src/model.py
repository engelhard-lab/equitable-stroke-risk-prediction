import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from sklearn.metrics import roc_auc_score


class Fair_DTFT(Model):
    
    def __init__(self, bins,
                 n_event_types=1,
                 encoder_layer_sizes=[256, ],
                 decoder_layer_sizes=[256, ],
                 activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-8):
        
        super(Fair_DTFT, self).__init__()

        self.bins = bins
        self.n_bins = len(bins) - 1

        self.n_event_types = n_event_types
        
        self.ld = ld
        self.lr = lr
        
        self.activation=activation
        self.tol = tol
        
        self.encoder_layers = [self.dense(ls) for ls in encoder_layer_sizes]
        self.decoder_layers = [self.dense(ls) for ls in decoder_layer_sizes]

        # self.prediction_head = [
        #     self.dense(self.n_bins + 1, activation='softmax')
        #     for i in range(n_event_types)
        # ]

        self.prediction_head = self.dense(
            self.n_bins * n_event_types + 1,
            activation='softmax'
        )
        
        self.encoder = Sequential(self.encoder_layers)
        self.decoder = Sequential(self.decoder_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=None
        )
        
        return(layer)
        
    
    def forward_pass(self, x):
        
        self.representation = self.encoder(x)
        self.features = self.decoder(self.representation)

        # self.t_pred = tf.stack(
        #     [ph(self.features) for ph in self.prediction_head],
        #     axis=-1
        # )

        self.t_pred = tf.reshape(
            self.prediction_head(self.features)[:, :-1],
            (-1, self.n_bins, self.n_event_types)
        )
        
        return self.t_pred
    
    
    def call(self, x):
        return self.forward_pass(x)


    def predict(self, x):
        return self.forward_pass(x)


    def predict_survival_function(self, x, t):

        y_complete_bins = get_proportion_of_bins_completed(
            np.ones(len(x)) * t, self.bins
        )

        return 1 - tf.reduce_sum(
            y_complete_bins[:, :, tf.newaxis] * self.forward_pass(x),
            axis=[1, 2]
        )
        
    
    def loss(self, x, t, s, mmd_binary_variable):

        nll = tf.reduce_mean(self.nll(x, t, s))
        l = nll

        # Add L2 regularizer

        for layer_l2 in self.losses:
            l += layer_l2
        
        # MMD Term
        l += self.ld * tf.cast(mmd(self.representation, mmd_binary_variable), dtype=tf.float32)
        
        return l, nll
    
    
    def nll(self, x, t, s):

        y = discretize_times(t, self.bins)        
        yt = tf.cast(y, dtype=tf.float32)

        y_complete_bins = tf.cast(
            get_proportion_of_bins_completed(t, self.bins),
            dtype=tf.float32
        )

        t_pred = self.forward_pass(x)

        ft = tf.reduce_sum(yt[:, :, tf.newaxis] * t_pred, axis=1) + self.tol
        
        Ft = 1 - tf.reduce_sum(
            y_complete_bins[:, :, tf.newaxis] * t_pred,
            axis=[1, 2]
        )[:, tf.newaxis] + self.tol

        # choose Ft if s is 0; otherwise pick correct ft
        likelihood = tf.reduce_sum(
            tf.concat([Ft, ft], axis=1) * tf.one_hot(s, self.n_event_types + 1),
            axis=1
        )
        
        return -1 * tf.math.log(likelihood)


def discretize_times(times, bins):

    bin_starts = np.array(bins[:-1])[np.newaxis, :]
    bin_ends = np.array(bins[1:])[np.newaxis, :]

    t = np.array(times)[:, np.newaxis]

    return ((t > bin_starts) & (t <= bin_ends)).astype(float)


def get_proportion_of_bins_completed(times, bins):

    bin_starts = np.array(bins[:-1])[np.newaxis, :]
    bin_ends = np.array(bins[1:])[np.newaxis, :]

    bin_lengths = bin_ends - bin_starts

    t = np.array(times)[:, np.newaxis]

    return np.maximum(np.minimum((t - bin_starts) / bin_lengths, 1), 0)


def mmd(x, s, beta=None):

    if beta is None:
        beta = get_median(tf.reduce_sum((x[:, tf.newaxis, :] - x[tf.newaxis, :, :]) ** 2, axis=-1))
    
    x0 = tf.boolean_mask(x, s == 0, axis=0)
    x1 = tf.boolean_mask(x, s == 1, axis=0)

    x0x0 = gaussian_kernel(x0, x0, beta)
    x0x1 = gaussian_kernel(x0, x1, beta)
    x1x1 = gaussian_kernel(x1, x1, beta)
    
    return tf.reduce_mean(x0x0) - 2. * tf.reduce_mean(x0x1) + tf.reduce_mean(x1x1)


def gaussian_kernel(x1, x2, beta=1.):
    return tf.exp(-1. * beta * tf.reduce_sum((x1[:, tf.newaxis, :] - x2[tf.newaxis, :, :]) ** 2, axis=-1))


def survival_from_density(f):
    return 1 - tf.math.cumsum(f, axis=1)
    #return tf.math.cumsum(f, reverse=True, axis=1)


def train_model(
    model, train_data, val_data, n_epochs,
    batch_size=50, learning_rate=1e-3, early_stopping_criterion=2,
    overwrite_output=True):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #@tf.function
    def train_step(x, t, s, mbv):
        with tf.GradientTape() as tape:
            train_loss, train_nll = model.loss(x, t, s, mbv)
            #print(train_loss, train_nll)
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return train_loss, train_nll

    #@tf.function
    def test_step(x, t, s, mbv):
        val_loss, val_nll = model.loss(x, t, s, mbv)
        return val_loss, val_nll
    
    best_val_loss = np.inf
    no_decrease = 0

    for epoch_idx in range(n_epochs):

        train_losses = []
        train_nlls = []

        for batch_idx, (xt, tt, st, mbvt) in enumerate(get_batches(*train_data, batch_size=batch_size)):

            #print(model(xt))

            train_loss, train_nll = train_step(xt, tt, st, mbvt)

            #print(train_loss)
            #print(train_nll)

            train_losses.append(train_loss)
            train_nlls.append(train_nll)

        # Display metrics at the end of each epoch.
        #print('Epoch training loss: %.4f, NLL = %.4f' % (np.mean(batch_losses), np.mean(batch_nll)))

        val_losses = []
        val_nlls = []

        # Run a validation loop at the end of each epoch.
        for batch_idx, (xv, tv, sv, mbvv) in enumerate(get_batches(*val_data, batch_size=batch_size)):

            val_loss, val_nll = test_step(xv, tv, sv, mbvv)

            val_losses.append(val_loss)
            val_nlls.append(val_nll)
            
        new_val_loss = np.mean(val_losses)

        if overwrite_output:
            print(
                'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
                % (epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls)),
                end='\r'
            )

        else:
            print(
                'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
                % (epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls))
            )
                
        if new_val_loss > best_val_loss:
            no_decrease += 1
        else:
            no_decrease = 0
            best_val_loss = new_val_loss
            
        if no_decrease == early_stopping_criterion:
            break

    if overwrite_output:
        print(
            'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
            % (epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls))
        )
        print('')

    return epoch_idx, np.mean(train_losses), np.mean(train_nlls), np.mean(val_losses), np.mean(val_nlls)


def get_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0]//2
    return tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)


def get_batches(*arrs, batch_size=1):
    l = len(arrs[0])
    for ndx in range(0, l, batch_size):
        yield (arr[ndx:min(ndx + batch_size, l)] for arr in arrs)

