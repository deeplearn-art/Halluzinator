import torch

def perlin_ms(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=False, device):

  def interp(t):
    return 3 * t**2 - 2 * t ** 3

  def perlin(width, height, scale=10, device):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

  out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]

  for i in range(1 if grayscale else 3):
    scale = 2 ** len(octaves)
    oct_width = width
    oct_height = height

    for oct in octaves:
      p = perlin(oct_width, oct_height, scale)
      out_array[i] += p * oct
      scale //= 2
      oct_width *= 2
      oct_height *= 2

  return torch.cat(out_array)

def generate_perlin(dim,grayscale=False): 
  octave_base = 1
  octave_length = 7
  perlin_width = 1
  perlin_height = 1
  #perlin_octaves = [octave_base**-(i) for i in range(octave_length)] # default
  perlin_octaves = [1/2, 1/4, 1/8, 1/16, 1/32, 1/48, 1/64]
  out = perlin_ms(perlin_octaves, perlin_width, perlin_height, grayscale)  
  if grayscale:
    out = transforms.Resize(dim)(out.unsqueeze(0))
    out = out.clamp(0, 1)
    pil = transforms.ToPILImage()(out).convert('RGB')
  else:  
    out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
    out = transforms.Resize((dim))(out)
    out = out.clamp(0, 1)
    pil = transforms.ToPILImage()(out.squeeze()).convert('RGB')
  return pil

def initial_img(option,dim,device):
    if option == "perlin":
        return generate_perlin(dim,device)
    elif option == "perlin grayscale":
        return generate_perlin(dim,device,grayscale=True)