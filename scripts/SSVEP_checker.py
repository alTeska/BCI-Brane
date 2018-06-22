import pyglet
import ratcave as rc
from itertools import cycle

window = pyglet.window.Window(resizable=True)
pyglet.clock.schedule(lambda dt: dt)

#initialize objects
obj_filename = rc.resources.obj_primitives

# new texture
# texture = rc.Texture.from_image(rc.resources.img_uvgrid)
texture = rc.Texture.from_image('img/uvgrid_bw.png')
# texture = rc.Texture.from_image('img/a.jpg')

plane_up = rc.WavefrontReader(obj_filename).get_mesh("Plane",  dynamic=True)
plane_left = rc.WavefrontReader(obj_filename).get_mesh("Plane",  dynamic=True)
plane_right = rc.WavefrontReader(obj_filename).get_mesh("Plane",  dynamic=True)

plane_up.texcoords *= .25
plane_right.texcoords *= .25
plane_left.texcoords *= .25


#set location and size of the objects
plane_up.position.xyz = 0, .8, -2 #location
plane_up.scale.xzy = .2, .2, .29 #size

plane_left.position.xyz = -1.1, -.8, -2
plane_left.scale.xzy = .2, .2, .29

plane_right.position.xyz = 1.1, -.8, -2
plane_right.scale.xzy = .2, .2, .29


blink_freq = 0 #the counter object that will be increased with screen refresh rate 60Hz

#function for toggling the color
def blink(df):
    global blink_freq
    if (blink_freq % 4) == 0: #15Hz
        plane_up.texcoords[:, 0] = 1 - plane_up.texcoords[:, 0]
        # plane_up.texcoords = 1 - plane_up.texcoords
    if (blink_freq % 6) == 0: #10 Hz
        plane_left.texcoords[:, 0] = 1 - plane_up.texcoords[:, 0]
        # plane_left.texcoords
    if (blink_freq % 3) == 0: #20Hz
        plane_right.texcoords[:, 0] = 1 - plane_up.texcoords[:, 0]


    blink_freq += 1

#function for displaying
@window.event
def on_draw():
    window.clear()
    with rc.default_shader, rc.default_states, texture:
        plane_up.draw()
        plane_left.draw()
        plane_right.draw()

#run
pyglet.clock.schedule(blink)
pyglet.app.run()
