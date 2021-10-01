
import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
from taming.models.cond_transformer import Net2NetTransformer

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, checkpoint_path):
  if config.model.target == 'taming.models.vqgan.VQModel':
    model = VQModel(**config.model.params)
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)
  elif config.model.target == 'taming.models.vqgan.GumbelVQ':
    model = GumbelVQ(**config.model.params)
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)
  elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
    parent_model = Net2NetTransformer(**config.model.params)
    parent_model.eval().requires_grad_(False)
    parent_model.init_from_ckpt(checkpoint_path)
    model = parent_model.first_stage_model
  del model.loss
  return model

class Vqgan(object):
  def __init__(self, cfg,ckpt,device):
    self.model = load_vqgan(load_config(cfg),ckpt)
    self.model.to(DEVICE)

  def _preprocess(self,x):
    x = 2.*x - 1.
    return x

  def decode(self,x):
    o_i3 = self.model.post_quant_conv(x)
    return self.model.decoder(o_i3)  
  
  def encode(self,im):
    im = self._preprocess(im)
    z, _, [_, _, _] = self.model.encode(im)
    return z
