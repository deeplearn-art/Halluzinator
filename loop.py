import torch
import numpy as np 
import torch.nn.functional as F

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
    
  def set_cam(self,cam):
    self.cam = cam  

  def set_options(self,opt):
    self.opt = opt  

  def print(self,str):
    if self.print_fn is not None:
      self.print_fn(str)  
    
  def display(self,img):
    if self.disp_fn is not None:
      self.disp_fn(img)  

  def run(self):
    count = 0
    self.opt.encs = self.encode_prompts()
    while(count < self.opt.total_count):
      loss = self.similarity().mean()
      self.print(f"loss {loss.item()}")  #TODO change to ui.console()
      self.losses.append(loss.item())
      self.gen.step(loss)
    
      if count % self.opt.burnin == 0:
        self.checkin()
      self.print(f"count {count} frames :{self.opt.frame_count}")  #TODO change to ui.console()
      count += 1  
    return self.opt 
  
  def similarity(self):
    im = self.format_image(self.gen())
    cutouts = self.cutout(im)
    enc_imgs = self.prc.encode_image(cutouts).requires_grad_() 
    sim =  10*-F.cosine_similarity(self.opt.encs, enc_imgs, -1)
    return sim
  
  def checkin(self):
    im_out = (self.gen().cpu().clip(-1, 1) + 1) / 2 
    im = im_out[0].detach().numpy()
    self.write_img(im) 
    self.opt.frame_count += 1
    if self.opt.count % self.opt.display_interval == 0:
      self.display('view.jpg') #TODO ui call
    elif self.opt.show_augs:
      augs_img = self.augment(alnot)
      augs_img = augs_img[0].cpu().detach().numpy()
      self.write_img(augs_img)
      self.display('augs.jpg')
  
  def rewind(self,frame): #TODO needs testing
    #print(f"{len(images)} images before edit")
    self.images = self.images[:frame+1]
    #print(f"{len(images)} images after edit")
    self.opt.frame_count = frame
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

  def write_img(self,img):
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
      self.camera(img)
      
  def camera(self,img):
    cam_off = all(m == 'off' for m in self.opt.moves) or self.opt.moves == []
    if self.opt.frame_count > 1 and not cam_off:  
      img = self.cam.move(img,self.opt.moves,self.opt.incs)
      self.gen.register(img,slerp_val=self.opt.slerp_val)

  def losses(self):
    return self.losses
   
  def encode_prompts(self):
    t = 0
    for i,prompt in enumerate(self.opt.prompts):
      enc = self.prc.encode_prompt(prompt)
      t += enc * self.opt.weights[i]
    return t
  