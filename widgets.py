
import ipywidgets as ipy

lo30 = ipy.Layout(width='30%')


noise_slider = ipy.FloatSlider(
    value=0.0,
    min=0.0,
    max=1.0,
    step=0.1,
    description='noise:',
    disabled=False,
    continuous_update=False,
    orientation='vertical',
    readout=True,
    readout_format='.1f',
)

decay_slider = ipy.FloatSlider(
    value=0.1,
    min=0.00,
    max=1.00,
    step=0.01,
    description='decay',
    disabled=False,
    continuous_update=False,
    orientation='vertical',
    readout=True,
    readout_format='.1f',
)
lr_slider = ipy.FloatSlider(
    value= 0.1,
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
    value=32,
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


mean_slider = ipy.FloatSlider( 
    value=0.35,
    min=0.00,
    max=1.00,
    step=0.01,
    description='mean',
    disabled=False,
    continuous_update=False,
    orientation='vertical',
    readout=True,
    readout_format='.2f',
)

std_slider = ipy.FloatSlider(
    value=0.80,
    min=0.00,
    max=1.00,
    step=0.01,
    description='std',
    disabled=False,
    continuous_update=False,
    orientation='vertical',
    readout=True,
    readout_format='.2f',
)

clamp_min_slider = ipy.FloatSlider(
    value=0.43,
    min=0.00,
    max=2.00,
    step=0.01,
    description='clamp_min',
    disabled=False,
    continuous_update=False,
    orientation='vertical',
    readout=True,
    readout_format='.2f',
)

clamp_max_slider = ipy.FloatSlider(
    value=1.90,
    min=0.00,
    max=4.00,
    step=0.01,
    description='clamp_max',
    disabled=False,
    continuous_update=False,
    orientation='vertical',
    readout=True,
    readout_format='.2f',
)

px_pf_slider = ipy.IntSlider(
    value=1,
    min=1,
    max=10,
    step=1,
    description='Movement speed',
    disabled=False,
    continuous_update=True,
    orientation='Horizontal',
    readout=True,
    readout_format='d'
)

sliders = ipy.HBox([noise_slider,lr_slider,decay_slider,cutn_slider])

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
transfer_chk = ipy.Checkbox(
    value=False,
    description='Transfer',
    disabled=False,
    indent=False
)
mp4_txt = ipy.Text(
    value = "",
    placeholder='',
    description='Use mp4',
    disabled=False
)

frame_h = ipy.HBox([frame_chk,frame_txt,transfer_chk, mp4_txt])
interval_txt = ipy.BoundedIntText(
    value= 20,
    min=0,
    max=5000,
    step=1,
    description='Interval',
    disabled=False
)
burnin_slider = ipy.IntSlider(
    value= 1,
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
    value=0.35,
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
    value='off',
    description='Move1',
    disabled=False
)
cam_dd_2 = ipy.Dropdown(
    options=options,
    value='off',
    description='Move2',
    disabled=False
)
px_pf_slider = ipy.IntSlider(
    value=1,
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
    value=1,
    min=1,
    max=30,
    step=1,
    description='speed2',
    disabled=False,
    continuous_update=True,
    orientation='Horizontal',
    readout=True,
    readout_format='d'
)
rotate_chk = ipy.Checkbox(
    value=False,
    description='Rotate',
    disabled=False,
    indent=False
)
angle_txt = ipy.BoundedIntText(
    value=0,
    min=0,
    max=360,
    step=1,
    description='Angle',
    disabled=False
)

rot = ipy.HBox([rotate_chk,angle_txt])

text_prompt_1 = ipy.Text(
    value = "",
    placeholder='',
    description='Prompt 1',
    disabled=False
)
rnd_chk_1 = ipy.Checkbox(
    value=False,
    description='random',
    disabled=False,
    indent=False
)
text_prompt_2= ipy.Text(
    value= "",
    placeholder='',
    description='Prompt 2',
    disabled=False
)
rnd_chk_2 = ipy.Checkbox(
    value=False,
    description='random',
    disabled=False,
    indent=False
)
chk_show_augs = ipy.Checkbox(
    value=False,
    description='Show augs',
    disabled=False,
    indent=False
)
text_topic= ipy.Text(
    value= "",
    placeholder='',
    description='img topic',
    disabled=False
)
weight_1 = ipy.BoundedFloatText(
    value = 1,
    min=-1.0,
    max=1.0,
    description='Weight1',
    disabled=False
)
weight_2 = ipy.BoundedFloatText(
    value = 0,
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

lo30 = ipy.Layout(width='30%')
lo50 = ipy.Layout(width='50%')
outpic = ipy.Output()
augput = ipy.Output()

left_pane = ipy.VBox([cam_dd,px_pf_slider,cam_dd_2,px_pf_slider_2,sliders])
center= ipy.HBox([outpic,left_pane,augput])
interval_value = 20

