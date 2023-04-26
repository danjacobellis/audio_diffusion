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

def σ_prior(band):
    def sc(z,μ,σ,γ):
        return sp.stats.skewcauchy.pdf(z, γ, loc=μ, scale=σ)
    return 10000*(2*sc(band,20,100,0.9)+sc(band,22,12,0.5))

def img_to_mdct(img):
    X = []
    q = 256;
    Y = pil_to_tf(img)
    Y = tf.cast(Y,tf.float32)/q
    for i_band in range(512):
        band = Y[:,i_band]
        σ = σ_prior(i_band)
        X.append(Finv(band,0,σ,0.85))
    X = tf.stack(X)
    X = tf.transpose(X)
    X = tf.where(tf.math.is_inf(X), tf.ones_like(X), X)
    return tf.cast(X,tf.float32)

def mdct_to_img(X):
    Y = []
    q = 256;
    for i_band in range(512):
        band = X[:,i_band]
        σ = σ_prior(i_band)
        Y.append(F(band,0,σ,0.85))
    Y = tf.stack(Y)
    Y = tf.transpose(Y)
    Y = tf.round(q*Y)
    Y = tf.cast(Y,tf.uint8)
    return tf_to_pil(Y)