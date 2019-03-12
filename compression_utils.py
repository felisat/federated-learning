from functools import partial
import torch
import numpy as np
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def approx_v(T, p, frac):
  if frac < 1.0:
    n_elements = T.numel()
    n_sample = min(int(max(np.ceil(n_elements * frac), np.ceil(100/p))), n_elements)
    n_top = int(np.ceil(n_sample*p))
    
    if n_elements == n_sample:
      i = 0
    else:
      i = np.random.randint(n_elements-n_sample)

    topk, _ = torch.topk(T.flatten()[i:i+n_sample], n_top)
    if topk[-1] == 0.0 or topk[-1] == T.max():
      return approx_v(T, p, 1.0)
  else:
    n_elements = T.numel()
    n_top = int(np.ceil(n_elements*p))
    topk, _ = torch.topk(T.flatten(), n_top)

  return topk[-1], topk 


def none(T, hp):
  '''
  Identity
  '''
  return T


def dgc(T, hp):
  '''
  "Deep Gradient Compression: Reducing the communication Bandwidth for Distributed Training, Lin et al."
  '''
  hp_ = {'p' : 0.001, 'approx' : 1.0}
  hp_.update(hp)
  
  if hp_['p'] >= 1.0:
    return T

  T_abs = torch.abs(T)
  
  v, _ = approx_v(T_abs, hp_["p"], hp_["approx"])
  
  out = torch.where(T_abs >= v, T, torch.Tensor([0.0]).to(device))
  
  return out


def stc(T, hp):
  '''
  "Sparse Binary Compression: Towards Distributed Deep Learning with minimal Communication, Sattler et al."
  '''
  hp_ = {'p' : 0.001, 'approx' : 1.0}
  hp_.update(hp)
  
  T_abs = torch.abs(T)

  v, topk = approx_v(T_abs, hp_["p"], hp_["approx"])
  mean = torch.mean(topk)  

  out_ = torch.where(T >= v, mean, torch.Tensor([0.0]).to(device))
  out = torch.where(T <= -v, -mean, out_)

  return out


def signsgd(T, hp):
  """
  signSGD: Compressed Optimisation for non-convex Problems, Bernstein et al.

  """
  return T.sign()


def compression_function(name, hp=None):
  '''
  Returns a function that maps a tensor to a tensor of the same shape
  '''
  return partial(globals()[name], hp=hp)





###############################################################################################
# COUNTING BITS
###############################################################################################


def get_bits(T, compression_method, approx=False):
  """
  Returns the number of bits that are required to communicate the Tensor T, which was compressed with compresion_method
  """

  B_val = {"none" : 32, "dgc" : 32, "stc" : 1, "signsgd" : 1}[compression_method]

  # dense methods
  if compression_method in ["none", "signsgd"]:
    k = T.numel()
    B_pos = 0

  # sparse methods non-optimal encoding
  elif compression_method in ["dgc"]:
    k = torch.sum(T!=0.0).item()
    B_pos = 16

  # sparse methods golomb encoding
  elif compression_method in ["stc"]:
    k = torch.sum(T!=0.0).item()
    N = T.numel()
    
    q = (k+1)/(N+1)
    golden = (np.sqrt(5)+1)/2

    if q == 1:
      return B_val*T.numel()
    if q == 0:
      return 0

    b_star = 1 + np.floor(np.log2(np.log(golden-1)/np.log(1-q)))

    if approx:
      B_pos = b_star + 1/(1-(1-q)**(2**b_star)) + 1
    else:
      idc = torch.nonzero(T.view(-1))
      distances = idc[:]-torch.cat([torch.Tensor([[-1]]).long().to("cuda"),idc[:-1]])
      B_pos = torch.mean(torch.ceil(distances.float()/2**b_star)).item()+(b_star+1)

  b_total = (B_pos+B_val)*k

  return b_total


def get_update_size(dW, compression_method):
  """
  Returns the number of bits that are required to communicate the entire network dW, which was compressed with compresion_method
  """

  update_size = sum([get_bits(T, compression_method[0]) for T in dW.values()])  

  return update_size



