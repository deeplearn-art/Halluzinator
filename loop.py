import torch
import numpy as np 
import torch.nn.functional as F
import random
import imageio

class Loop(object):
  def __init__(self, prc, gen, device, augs, print_fn=None, disp_fn=None):
    self.device = device
    self.gen = gen
    self.prc = prc
    self.augs = augs
    self.print_fn = print_fn
    self.disp_fn = disp_fn
    self.cam = None
    self.opt = None
    self.losses = []
    self.images = []
    self.denoise = 0.5
   
  def set_cam(self,cam):
    self.cam = cam  

  def set_options(self,opt):
    self.encs = []
    for prompt in opt.prompts:
      self.encs.append(self.prc.encode_prompt(prompt))
    self.opt = opt    
      
  def print(self,str):
    if self.print_fn is not None:
      self.print_fn(str)  
    
  def display(self,img):
    if self.disp_fn is not None:
      self.disp_fn(img)  

  def run(self):
    count = 0
    while(count < self.opt.total_count):
      im = self.format_image(self.gen())
      
      loss_t = self.clip_loss(im)
      loss_t += self.lossTV(im,self.opt.denoise)
      loss = loss_t.mean()
      self.print(f"loss {loss.item()}")  #TODO change to ui.console()
      self.losses.append(loss.item())
      self.gen.step(loss)
    
      if count % self.opt.burnin == 0:
        self.checkin()
      
      self.print(f"count {count} frames :{self.opt.frame_count}")  #TODO change to ui.console()
      count += 1  
    return self.opt 
  
  def clip_loss(self, im):
    cutouts = self.cutout(im)
    enc_imgs = self.prc.encode_image(cutouts).requires_grad_() 
    total_loss = 0
    for i,enc in enumerate(self.encs):
      sim = F.cosine_similarity(enc, enc_imgs, -1)
      w = self.opt.weights[i]
      if w > 0:
        loss = w * (1 - sim)
      else:
        loss = -1 * w * sim  
      total_loss += loss
    return total_loss

  def lossTV(self,image, denoise):
    Y = (image[:,:,1:,:] - image[:,:,:-1,:]).abs().mean()
    X = (image[:,:,:,1:] - image[:,:,:,:-1]).abs().mean()
    loss = (X + Y) * 0.5 * denoise
    return loss  
  
  def checkin(self):
    im_out = (self.gen().clip(-1, 1) + 1) / 2 
    im_np = im_out[0].cpu().detach().numpy()
    self.write_img(im_np,im_out) 
    self.opt.frame_count += 1
    if self.opt.count % self.opt.display_interval == 0:
      self.display('view.jpg') #TODO ui call
    elif self.opt.show_augs:
      augs_img = self.augment(alnot)
      augs_img = augs_img[0].cpu().detach().numpy()
      self.write_img(augs_img)
      self.display('augs.jpg')
  
  def undo(self,interval): #TODO needs testing
    #print(f"{len(images)} images before edit")
    self.images = self.images[:-interval]
    #print(f"{len(images)} images after edit")
    self.opt.frame_count -= interval
    self.gen.register(self.images[-1])   

  def cutout(self,into):
    sideY, sideX = into.shape[2:4]
    min_dim = min(sideX, sideY)
    p_size = self.prc.size
    
    cutouts = torch.zeros(self.opt.cutn,3,p_size,p_size)
    for n in range(self.opt.cutn):
      img = self.augs(into)
      if min_dim < p_size - 32:
        offset = min_dim
      else:  
        offset = random.randint(p_size - 32, min_dim - 1)

      oy = random.randrange(0, sideY-offset) 
      ox = random.randrange(0, sideX-offset)
      img = img[ : , : , oy : oy+offset, ox : ox+offset ] 
      img = F.interpolate(img, size=(p_size,p_size), mode='bilinear', align_corners=False)
      cutouts[n] = img

    cutouts += self.opt.noise * torch.randn_like(cutouts, requires_grad=False)
    return cutouts
  
  def format_image(self,x):
    x = torch.tanh(x+x**5*0.5)
    x = (x + 1.)/2
    return x

  def write_img(self,img,im):
    #img = img.detach().numpy()
    img = np.array(img)[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    self.images.append(img)
  
    img = (img*255).astype(np.uint8)
    if self.opt.show_augs:
      imageio.imwrite('augs.jpg', img)
      return
    imageio.imwrite('view.jpg', img)
    imageio.imwrite(f"{self.opt.save_path}/{self.opt.frame_count:05}.jpg", np.array(img))
    if (self.cam != None):
      self.camera(img,im)
      
  def camera(self,img,im):
    cam_off = all(m == 'off' for m in self.opt.moves) or self.opt.moves == []
    if self.opt.frame_count > 1 and not cam_off:  
      img = self.cam.move(img,self.opt.moves,self.opt.incs)
    if (DepthStrength > 0):       
      im = depth_transform(im, img, depth_infer, depth_mask, size, DepthStrength,scale=0.97,shift=[-2,0])    
      self.gen.register(im,slerp_val=self.opt.slerp_val,is_numpy=False)
    else:
      self.gen.register(img,slerp_val=self.opt.slerp_val)  
