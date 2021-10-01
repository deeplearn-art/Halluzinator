import torch
from torchvision import transforms

class Generator(torch.nn.Module):
  def __init__(self, model, optimizer, dim, device):
    super(Generator, self).__init__()
    self.model = model
    self.optimizer = optimizer
    self.device = device
    self.dim = dim
    self.latent = 0

  def register(self,img,slerp_val=0,is_numpy=True):
    with torch.no_grad():
      if is_numpy:
        im = self.np2tensor(img)
      im = transforms.Resize(self.dim)(im)
      im = im.to(self.device)
      z = self.encode(im)
      if slerp_val > 0: 
        z = self.slerp(self.latent, z, slerp_val)       
      self.latent = torch.nn.Parameter(z.requires_grad_(True))
      self.optimizer.param_groups[0]['params'][:] = [self.latent]
  
  def np2tensor(self,img):
    return torch.unsqueeze(transforms.ToTensor()(img), 0)   

  def encode(self,img):
    return self.model.encode(img)

  def slerp(self,low, high, val,epsilon=1e-7):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = (low_norm*high_norm).sum(1)
    omega = torch.acos(torch.clamp(omega, -1 + epsilon, 1 - epsilon))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res  

  def step(self,loss):
    with torch.enable_grad():
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      

  def forward(self):
    return self.model.decode(self.latent).to(self.device)
  