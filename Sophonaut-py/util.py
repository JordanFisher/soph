import numpy as np
import matplotlib.pyplot as pl

def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
       raise ValueError, "`window` must be at least 1."
    if window > a.shape[-1]:
       raise ValueError, "`window` is too long."
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window(a, window):
    """Takes a numpy array *a* and a sequence of (or single) *window* lengths
    and returns a view of *a* that represents a moving window."""
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a

def norm2(v):
    return np.dot(v, v)

def norm(v):
    return norm2(v) ** .5

def normalized(v):
    return v / norm(v)

def similiarty(w, v):
    dot = np.dot(w, v)
    return abs(dot) / (norm(w) * norm(v))

def error(w, v):
    n = norm2(v)
    d = np.dot(w, v)

    return (n - d*d) / (n + .00001)

def list_norm2(l):
    return (l * l).sum(1)

def list_norm(l):
    return list_norm2(l) ** .5

def list_normalized(l):
    norm = list_norm(l)
    norm.shape = (norm.shape[0], 1)

    return l / (norm + .00001)

def list_similarity_prenormalized(l, w):
    dot = np.dot(l, w)
    return abs(dot)

def list_similarity(l, w):
    dot = np.dot(l, w)
    return abs(dot) / (list_norm(l) * norm(w) + .00001)

def list_error(l, w):
    n2 = list_norm2(l)
    d = np.dot(l, w)
    
    return (n2 - d*d) / (n2 + .00001)
