import tensorflow as tf
import tensorflow_io as tfio
import IPython.display as ipd
import matplotlib.pyplot as plt
import scipy as sp
import PIL.Image
import numpy as np

def wav_to_tf(filename):
    bits = tf.io.read_file(filename)
    x = tfio.audio.decode_wav(bits,dtype=tf.int16)[:,0]
    x = tf.cast(x,tf.float32)
    x = x - tf.math.reduce_mean(x);
    x = x / tf.math.reduce_std(x)
    return tf.Variable(x)

def play(x,rate=24000):
    ipd.display(ipd.Audio(x,rate=rate,autoplay=False))

def slog(x):
    return tf.sign(x) * tf.math.log(1+ tf.math.abs(x) )
    
def show(X,clim=(-3,3), xlim=(0,300), ylim=(0,100)):
    plt.figure(figsize=(15,6),dpi=200)
    plt.imshow(tf.transpose(X),origin='lower',cmap='RdBu')
    plt.colorbar()
    plt.clim(clim)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
def mdct(x,L=624):
    X = tf.signal.mdct(x,L);
    return tf.Variable(X)

def imdct(X):
    y = tf.signal.inverse_mdct(X)
    return y

Γ = sp.special.gamma

def F(x,μ,σ,γ):
    return sp.stats.gennorm.cdf(x, beta=γ, loc=μ, scale=σ)

def Finv(x,μ,σ,γ):
    return sp.stats.gennorm.ppf(x, beta=γ, loc=μ, scale=σ)

def r(γ):
    return Γ(1/γ)*Γ(3/γ)/Γ(2/γ)

def estimate_GGD(X):
    μ = tf.math.reduce_mean(X)
    σ = tf.math.reduce_std(X)
    E = tf.math.reduce_mean(tf.abs(X - μ))
    ρ = tf.square(σ/E)
    
    γ = sp.optimize.bisect(lambda γ:r(γ)-ρ, 0.3, 1.5,maxiter=50)
    return μ,σ,γ

def tf_to_pil(x):
    x = np.array(x)
    return PIL.Image.fromarray(x,mode="L")
def pil_to_tf(x):
    x = np.array(x)
    return tf.convert_to_tensor(x)