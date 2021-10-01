
import clip
import torch
import PIL
import torch.nn.functional as F
from torchvision import transforms

class Perceptor(object):
  def __init__(self, device):
    self.device = device
    self.model = None
    self.size = 0
    self.nom = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

  def load(self,clip_model='ViT-B/16'):
    self.model, preprocess = clip.load(clip_model,jit=False)
    self.model.eval().eval().requires_grad_(False);
    self.size = preprocess.transforms[0].size   

  def size(self):
    return self.size

  def encode_prompt(self,txt):
    if '/' in txt:
      img = PIL.Image.open(txt)
      return self.encode_target_img(img)
    else:
      tx = clip.tokenize(txt).to(self.device)
      return self.model.encode_text(tx).detach().clone()

  def encode_target_img(self,img):
    im = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
    im = (F.interpolate(im,(self.size,self.size)) / 255).to(self.device)[:,:3]
    return self.encode_image(im)

  def encode_image(self,im):
    img_enc = self.nom(im).to(self.device)
    return self.model.encode_image(img_enc)
