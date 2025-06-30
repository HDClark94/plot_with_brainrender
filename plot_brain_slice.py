import brainglobe_heatmap as bgh

values = dict(  # scalar values for each region
    ENTm=1,
    VIS=3,
    PRE=2,
    PAR=4,
    HIP=5,
    SUB=6,
)

import brainglobe_heatmap as bgh

position = (10300,
            5000,
            5000
            )
orientation = (1,
               -0.15,
               0
               )

planner = bgh.plan(
    values,
    position=position,
    orientation=orientation,  # orientation, or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    thickness=200,  # thickness of the slices used for rendering (in microns)
).show()

f = bgh.Heatmap(
    values,
    position=position,
    orientation=orientation,
    thickness=1000,
    atlas_name="allen_mouse_10um",
    format='2D',
    cmap='Set2'
    ).show(filename='/Users/harryclark/Downloads/probe_face_view.pdf')
    

f = bgh.Heatmap(
    values,
    position=2500,
    orientation="sagittal",  # 'frontal' or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    thickness=1000,
    atlas_name="allen_mouse_10um",
    format='2D',
    cmap='Set2'
    ).show(filename='/Users/harryclark/Downloads/sagittal_view.pdf')
