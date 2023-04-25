import tensorflow as tf
import tensorflow_io as tfio
import IPython.display as ipd
import matplotlib.pyplot as plt

def wav_to_tf(filename):
    bits = tf.io.read_file(filename)
    x = tfio.audio.decode_wav(bits,dtype=tf.int16)[:,0]
    x = tf.cast(x,tf.float32)
    x = x - tf.math.reduce_mean(x);
    x = x / tf.math.reduce_std(x)
    return tf.Variable(x)

def play(x):
    ipd.display(ipd.Audio(x,rate=24000,autoplay=False))

def slog(x):
    return tf.sign(x) * tf.math.log(1+ tf.math.abs(x) )
    
def show(X,clim=(-3,3), xlim=(100,300), ylim=(0,100)):
    plt.figure(figsize=(15,6),dpi=200)
    plt.imshow(tf.transpose(X),origin='lower',cmap='RdBu')
    plt.colorbar()
    plt.clim(clim)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
def mdct(x):
    X = tf.signal.mdct(x,624);
    return tf.Variable(X)

def imdct(X):
    y = tf.signal.inverse_mdct(X)
    return y