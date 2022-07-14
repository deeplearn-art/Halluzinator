import ipywidgets as ipy

default_opt = {
"noise": 0.00, 
"slerp_val": 0.35, 
"burnin": 1, 
"lr": 0.1, 
"denoise": 0.5, 
"cutn": 32, 
"moves": ["off", "off"], 
"incs": [1, 1], 
"prompts": ["", ""],
"weights": [1.0, 0.0], 
"frame_count": 0, 
"save_path": "/content/frames", 
"show_augs": False, 
"display_interval": 1, 
"total_count": 0, 
"count": 0
}
def UI(opt=default_opt,interval=20):

  noise_slider = ipy.FloatSlider(
      value=opt["noise"],
      min=0.00,
      max=0.20,
      step=0.01,
      description='noise:',
      disabled=False,
      continuous_update=False,
      orientation='vertical',
      readout=True,
      readout_format='.2f',
  )
  #TODO?
  denoise_slider = ipy.FloatSlider(
      value=opt["denoise"],
      min=0.00,
      max=1.00,
      step=0.01,
      description='denoise',
      disabled=False,
      continuous_update=False,
      orientation='vertical',
      readout=True,
      readout_format='.1f',
  )
  lr_slider = ipy.FloatSlider(
      value=opt["lr"],
      min=0.000,
      max=0.500,
      step=0.001,
      description='lr',
      disabled=False,
      continuous_update=False,
      orientation='vertical',
      readout=True,
      readout_format='.3f',
  )
  cutn_slider = ipy.IntSlider(
      value=opt["cutn"],
      min=0,
      max=128,
      step=4,
      description='cutn',
      disabled=False,
      continuous_update=False,
      orientation='vertical',
      readout=True,
      readout_format='d'
  )

  sliders = ipy.HBox([noise_slider,lr_slider,denoise_slider,cutn_slider])

  frame_txt = ipy.BoundedIntText(
      value=0,
      min=0,
      max=999999999999,
      step=1,
      description='Frame start',
      disabled=False
  )
  frame_chk = ipy.Checkbox(
      value=False,
      description='Use frame',
      disabled=False,
      indent=False
  )
  

  frame_h = ipy.HBox([frame_chk,frame_txt])
  interval_txt = ipy.BoundedIntText(
      value= interval,
      min=0,
      max=5000,
      step=1,
      description='Interval',
      disabled=False
  )
  burnin_slider = ipy.IntSlider(
      value= opt["burnin"],
      min=1,
      max=25,
      step=1,
      description='Burn',
      disabled=False,
      continuous_update=True,
      orientation='Horizontal',
      readout=True,
      readout_format='d'
  )
  slerp_slider = ipy.FloatSlider(
      value=opt["slerp_val"],
      min=0.00,
      max=1.00,
      step=0.01,
      description='slerp',
      disabled=False,
      continuous_update=False,
      orientation='Horizontal',
      readout=True,
      readout_format='.2f',
  )

  frame_controls = ipy.HBox([interval_txt,ipy.VBox([burnin_slider, slerp_slider])])

  v = ipy.VBox(children=[frame_h,frame_controls])

  options=['off', 'warp','zoom_in', 'zoom_out', 'pan_left', 'pan_right','pan_up','pan_down','rotate']
  cam_dd = ipy.Dropdown(
      options=options,
      value=opt["moves"][0],
      description='Move1',
      disabled=False
  )
  cam_dd_2 = ipy.Dropdown(
      options=options,
      value=opt["moves"][1],
      description='Move2',
      disabled=False
  )
  px_pf_slider = ipy.IntSlider(
      value=opt["incs"][0],
      min=1,
      max=30,
      step=1,
      description='speed1',
      disabled=False,
      continuous_update=True,
      orientation='Horizontal',
      readout=True,
      readout_format='d'
  )
  px_pf_slider_2 = ipy.IntSlider(
      value=opt["incs"][1],
      min=-30,
      max=30,
      step=1,
      description='speed2',
      disabled=False,
      continuous_update=True,
      orientation='Horizontal',
      readout=True,
      readout_format='d'
  )

  text_prompt_1 = ipy.Text(
      value = opt["prompts"][0],
      placeholder='',
      description='Prompt 1',
      disabled=False
  )
  text_prompt_2= ipy.Text(
      value= opt["prompts"][1],
      placeholder='',
      description='Prompt 2',
      disabled=False
  )

  chk_show_augs = ipy.Checkbox(
      value=False,
      description='Show augs',
      disabled=False,
      indent=False
  )

  weight_1 = ipy.BoundedFloatText(
      value=opt["weights"][0],
      min=-1.0,
      max=1.0,
      description='Weight1',
      disabled=False
  )
  weight_2 = ipy.BoundedFloatText(
      value=opt["weights"][1],
      min=-1.0,
      max=1.0,
      description='Weight2',
      disabled=False
  )

  prompt1_h = ipy.HBox([text_prompt_1,weight_1])
  prompt2_h = ipy.HBox([text_prompt_2,weight_2])

  console = ipy.Output()
  console.layout.width='400px'
  console.layout.height='200px'

  outpic = ipy.Output()

  left_pane = ipy.VBox([cam_dd,px_pf_slider,cam_dd_2,px_pf_slider_2,sliders])
  center= ipy.HBox([outpic,left_pane])
  
  return (center,v,chk_show_augs, prompt1_h, prompt2_h, console, outpic,
  frame_txt,frame_chk,interval_txt,        
  noise_slider,slerp_slider,burnin_slider,lr_slider,denoise_slider,px_pf_slider,px_pf_slider_2,
  cutn_slider,cam_dd,cam_dd_2,text_prompt_1,text_prompt_2,weight_1,weight_2,chk_show_augs)
 

