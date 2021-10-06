import cv2
import numpy as np
from torchvision import transforms
import PIL

class Camera(object):
  def __init__(self,dim,resample=None):
    self.dim = dim
    self.resample = resample # PIL.Image.BILINEAR
    
  def move(self,img,moves,incs):
    for i,m in enumerate(moves):
      img = self.move_cam(img,m,incs[i])
    return img

  def move_cam(self,img,m,inc):
    if m == 'off':
      pass
    elif m == 'rotate':
      img = self.rotate_img(img,inc)  
    elif m == 'warp':
      img = self.warp(img,inc)    
    elif m == 'zoom_in':
      img = self.zoom_in(img,inc) 
    elif m == 'zoom_out':
      img = self.zoom_out(img,inc)  
    elif m[:3] == 'pan':
      img = self.pan(img,m,inc)
    return img    

  def crop(self,img, dx, dy, h, w):
    return img[dy:dy+h, dx:dx+w]

  def centre_crop(self,img,inc):
    h,w = self.dim
    d = max(2,inc//2)
    return img[d: h - d, d: w - d]  

  def get_new_dims(self,change_in_width):
    h,w = self.dim
    ratio = (w + change_in_width) / w
    return (w + change_in_width, int(h * ratio))

  def warp(self,img,inc):  
    h,w = self.dim
    pts1 = np.float32([[0, 0], [inc, h - inc], [w - inc, h - inc], [w, 0]])
    pts2 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return result  

  def zoom_in(self,img,inc):
    h,w = self.dim
    new_w,new_h = self.get_new_dims(inc)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LANCZOS4)
    return self.centre_crop(img,inc)
    
  def zoom_out(self,img,inc):  
    bdr = cv2.copyMakeBorder(img,inc,inc,inc,inc,cv2.BORDER_REPLICATE)
    return cv2.resize(bdr, dsize=(self.dim), interpolation=cv2.INTER_CUBIC)

  def pan(self,img,move,inc):
    h,w = self.dim
    ret = img
    if move == 'pan_left':
      # cv2.copyMakeBorder(im=img,top=0,bottom=inc,left=0,right=0,cv2.BORDER_REPLICATE)
      border = cv2.copyMakeBorder(img,0,inc,0,0,cv2.BORDER_REPLICATE)
      border_crop = self.crop(border, 0, 0, h, inc) # crop(img, dx, dy, h, w): img[dy:dy+h, dx:dx+w]
      img_crop = self.crop(img, 0, 0, h, w - inc)
      ret = np.concatenate((border_crop,img_crop), axis = 1)
    elif move == 'pan_up':
      border = cv2.copyMakeBorder(img,inc,0,0,0,cv2.BORDER_REPLICATE)
      border_crop = self.crop(border, 0, 0, inc, w)
      img_crop =self.crop(img, 0, 0,h - inc, w)
      ret = np.concatenate((border_crop,img_crop))
    elif move == 'pan_down':
      border = cv2.copyMakeBorder(img,0,0,inc,0,cv2.BORDER_REPLICATE)
      border_crop = self.crop(border, 0, inc, inc, w)
      img_crop = self.crop(img, 0, inc ,h -inc, w)
      ret = np.concatenate((img_crop, border_crop))
    elif move == 'pan_right': 
      border = cv2.copyMakeBorder(img,0,0,0,inc,cv2.BORDER_REPLICATE)
      #border = blur(border)
      border_crop =self.crop(border, h - inc, 0, h, inc)
      img_crop =self.crop(img, inc, 0, h, w - inc)
      ret = np.concatenate((img_crop, border_crop), axis = 1)
    return ret  

  def rotate_img(self,img,inc):
    padding = int(max(self.dim)/4) 
    #PIL_img = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
    pil = transforms.ToPILImage()(img).convert('RGB')
    img = transforms.functional.pad(img=img, padding=padding, padding_mode='reflect')
    img = transforms.functional.rotate(img, -inc, resample=self.resample)
    img = transforms.functional.crop(img, padding, padding, sideH, sideW)
    return np.asarray(img)
  

