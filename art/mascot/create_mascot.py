"""Original Student Support mascot. Run with Blender 4.5+ --background --python this_file.
Also runs with the official bpy Python module. All geometry is created here.
"""
from pathlib import Path
import math
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'assets'
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def material(name, hexcolor, roughness=.72):
    h = hexcolor.lstrip('#')
    rgb = [int(h[i:i+2],16)/255 for i in (0,2,4)]
    linear = [c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in rgb]
    m=bpy.data.materials.new(name); m.diffuse_color=(*linear,1); m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF'); p.inputs['Base Color'].default_value=(*linear,1)
    p.inputs['Roughness'].default_value=roughness
    return m
skin=material('Warm porcelain skin','#f2c6ab')
hair=material('Chestnut hair','#553d35')
hairlight=material('Chestnut highlights','#715246')
cream=material('Oatmeal cardigan','#e8ddc5')
trim=material('Ribbed knit trim','#cfc2a6')
sage=material('Sage green','#828e77')
sagedark=material('Deep sage','#5e6f5d')
clay=material('Terracotta notebook','#b66d50')
paper=material('Ivory pages','#fff4db')
dark=material('Soft brown linework','#392b2a')
iris=material('Amber eyes','#89573c',.4)
white=material('Eye whites','#fff6e7',.5)
blush=material('Rose cheeks','#dc9385')
shoe=material('Cocoa shoes','#5d483f')
gold=material('Warm brass','#b89054',.45)

def smooth(obj,mat):
    obj.data.materials.append(mat)
    if obj.type=='MESH':
        for p in obj.data.polygons:p.use_smooth=True
    return obj

def sphere(name,loc,scale,mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=32,location=loc)
    o=bpy.context.object; o.name=name; o.scale=scale
    return smooth(o,mat)

def rounded(name,loc,size,mat,bevel=.08):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc)
    o=bpy.context.object; o.name=name; o.dimensions=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    b=o.modifiers.new('Soft corners','BEVEL'); b.width=bevel;b.segments=5
    o.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
    return smooth(o,mat)

def curve(name,pts,radius,mat):
    c=bpy.data.curves.new(name,'CURVE'); c.dimensions='3D'; c.resolution_u=24
    c.bevel_depth=radius;c.bevel_resolution=4
    s=c.splines.new('BEZIER');s.bezier_points.add(len(pts)-1)
    for b,p in zip(s.bezier_points,pts):
        b.co=p;b.handle_left_type='AUTO';b.handle_right_type='AUTO'
    o=bpy.data.objects.new(name,c);bpy.context.collection.objects.link(o);o.data.materials.append(mat)
    return o

def capsule(name,a,b,width,mat,depth=None):
    mid=(Vector(a)+Vector(b))/2;length=(Vector(b)-Vector(a)).length
    o=sphere(name,mid,(width,depth or width,length/2+width*.45),mat)
    o.rotation_euler=(Vector(b)-Vector(a)).to_track_quat('Z','Y').to_euler()
    return o

# A compact, fully clothed chibi silhouette.
for side in (-1,1):
    x=side*.24
    capsule('Stocking', (x,0,.35),(x,0,1.05),.155,sagedark)
    sphere('Rounded loafer',(x,-.13,.23),(.235,.36,.18),shoe)
    curve('Shoe seam',[(x-.15,-.38,.28),(x,-.43,.31),(x+.15,-.38,.28)],.012,trim)
    rounded('Shoe strap',(x,-.31,.34),(.31,.14,.055),shoe,.025)

# Pleated skirt, a single closed mesh with alternating folds.
verts=[];faces=[];N=64
for z,r in [(1.02,.57),(1.13,.57),(1.62,.41)]:
    for i in range(N):
        a=2*math.pi*i/N;rr=r*(1+.042*math.cos(i*math.pi/2))
        verts.append((rr*math.cos(a),rr*.68*math.sin(a),z))
for j in range(2):
    for i in range(N):faces.append((j*N+i,j*N+(i+1)%N,(j+1)*N+(i+1)%N,(j+1)*N+i))
faces.extend([tuple(reversed(range(N))),tuple(range(2*N,3*N))])
mesh=bpy.data.meshes.new('Skirt folds');mesh.from_pydata(verts,[],faces);mesh.update()
o=bpy.data.objects.new('Sage pleated skirt',mesh);bpy.context.collection.objects.link(o);smooth(o,sage)
b=o.modifiers.new('Soft hem','BEVEL');b.width=.025;b.segments=3
o.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
sphere('Cardigan body',(0,0,1.98),(.54,.34,.66),cream)
rounded('Cardigan hem',(0,-.015,1.51),(1.03,.70,.17),trim,.06)
sphere('Neck',(0,0,2.59),(.19,.19,.30),skin)
# Collar and cardigan center stitching.
for side in (-1,1):
    o=rounded('Soft collar',(side*.15,-.30,2.44),(.25,.095,.25),paper,.055);o.rotation_euler.y=side*.40
curve('Cardigan opening',[(0,-.332,2.38),(0,-.353,2.0),(0,-.31,1.54)],.013,trim)
for z in (2.23,1.99,1.75):sphere('Wood button',(.04,-.364,z),(.034,.017,.034),gold)

# One hand holds a book; the other gives a small, welcoming wave.
capsule('Left sleeve',(-.41,0,2.35),(-.64,-.13,1.85),.22,cream)
capsule('Left forearm',(-.65,-.13,1.87),(-.34,-.48,1.78),.19,cream)
capsule('Right upper sleeve',(.42,0,2.34),(.79,-.02,2.38),.22,cream)
capsule('Raised sleeve',(.80,-.02,2.4),(1.04,-.04,2.91),.19,cream)
capsule('Raised cuff',(1.02,-.04,2.82),(1.07,-.04,2.96),.177,trim)
sphere('Waving palm',(1.09,-.035,3.12),(.145,.095,.18),skin)
for i,(x,z,length) in enumerate([(0.985,3.26,.16),(1.07,3.30,.21),(1.155,3.29,.19),(1.235,3.24,.14)]):
    capsule('Wave finger '+str(i),(x,-.035,z-.07),(x+.025,-.03,z+length*.65),.038,skin)
capsule('Wave thumb',(1.00,-.055,3.08),(.88,-.09,3.20),.054,skin)

book=bpy.data.objects.new('Notebook assembly',None);bpy.context.collection.objects.link(book)
book.location=(-.18,-.49,1.99);book.rotation_euler.y=-.17
for name,loc,size,mat,bv in [
    ('Notebook pages',(0,0,0),(.59,.105,.72),paper,.024),
    ('Notebook front',(0,-.079,0),(.66,.05,.79),clay,.035),
    ('Notebook back',(0,.079,0),(.66,.04,.79),clay,.035),
    ('Notebook spine',(-.304,0,0),(.065,.19,.79),clay,.025),
    ('Notebook label',(.01,-.108,.12),(.39,.012,.25),paper,.02),
    ('Bookmark',(.19,-.01,-.44),(.045,.04,.13),sage,.008)]:
    o=rounded(name,loc,size,mat,bv);o.parent=book
for i,h in enumerate([.055,.09,.13]):
    o=rounded('Notebook progress mark',(-.085+i*.085,-.121,.09+h/2),(.04,.014,h),sage,.009);o.parent=book
sphere('Holding hand',(-.48,-.635,1.83),(.15,.10,.12),skin)
for i in range(3):
    capsule('Holding fingers',(-.51+i*.058,-.705,1.82),(-.5+i*.058,-.704,1.90),.027,skin)

# Head and bob. Hair behind the face stays sculptural, with a clean outer edge.
sphere('Chestnut bob back',(0,.10,3.53),(.90,.65,.98),hair)
for side in (-1,1):
    o=sphere('Bob side',(side*.70,.02,3.17),(.25,.48,.58),hair);o.rotation_euler.y=side*.08
sphere('Face',(0,-.12,3.49),(.76,.59,.82),skin)
for side in (-1,1):
    sphere('Ear',(side*.746,-.13,3.38),(.105,.105,.155),skin)
    sphere('Ear detail',(side*.777,-.216,3.38),(.038,.02,.07),blush)

# A shaped cap leaves the forehead and cheeks open.
verts=[];faces=[];rows=24;cols=80
for j in range(rows+1):
    t=j/rows
    for i in range(cols):
        a=2*math.pi*i/cols
        # Front is negative Y. Lower edge rises to form the fringe.
        front=max(0,-math.sin(a))
        end=1.80-.57*front
        theta=.001+(end-.001)*t
        verts.append((.875*math.sin(theta)*math.cos(a),.06+.675*math.sin(theta)*math.sin(a),3.54+.98*math.cos(theta)))
for j in range(rows):
    for i in range(cols):faces.append((j*cols+i,j*cols+(i+1)%cols,(j+1)*cols+(i+1)%cols,(j+1)*cols+i))
mesh=bpy.data.meshes.new('Hair cap mesh');mesh.from_pydata(verts,[],faces);mesh.update()
o=bpy.data.objects.new('Sculpted hair cap',mesh);bpy.context.collection.objects.link(o);smooth(o,hair)
solid=o.modifiers.new('Hair thickness','SOLIDIFY');solid.thickness=.035
sub=o.modifiers.new('Smooth hair','SUBSURF');sub.levels=1

# Tapered curved locks, with elliptical cross-sections and pointed ends.
def lock(name,points,widths,mat):
    v=[];f=[];sides=16
    for k,(p,w) in enumerate(zip(points,widths)):
        for j in range(sides):
            a=2*math.pi*j/sides
            v.append((p[0]+w*math.cos(a),p[1]+w*.43*math.sin(a),p[2]))
    for k in range(len(points)-1):
        for j in range(sides):f.append((k*sides+j,k*sides+(j+1)%sides,(k+1)*sides+(j+1)%sides,(k+1)*sides+j))
    f.extend([tuple(reversed(range(sides))),tuple(range((len(points)-1)*sides,len(points)*sides))])
    m=bpy.data.meshes.new(name);m.from_pydata(v,[],f);m.update();o=bpy.data.objects.new(name,m);bpy.context.collection.objects.link(o);smooth(o,mat)
    sub=o.modifiers.new('Sculpted lock','SUBSURF');sub.levels=2
lock('Swept fringe left',[(-.10,-.47,4.32),(-.28,-.59,4.18),(-.40,-.68,3.95),(-.49,-.68,3.72),(-.53,-.61,3.54)],[.10,.22,.23,.16,.005],hair)
lock('Swept fringe center',[(.03,-.49,4.31),(.03,-.64,4.15),(.15,-.73,3.94),(.29,-.74,3.78),(.39,-.65,3.68)],[.05,.20,.23,.15,.004],hair)
lock('Swept fringe right',[(.15,-.45,4.29),(.37,-.57,4.12),(.52,-.62,3.87),(.63,-.60,3.61),(.66,-.47,3.41)],[.035,.17,.17,.11,.006],hair)
for side in (-1,1):
    lock('Face framing hair',[(side*.68,-.36,3.96),(side*.76,-.41,3.63),(side*.79,-.39,3.24),(side*.71,-.37,2.98)],[.10,.135,.11,.004],hair)
    curve('Fine hair strand',[(side*.79,-.12,3.86),(side*.86,-.24,3.45),(side*.81,-.23,3.10)],.010,hairlight)
curve('Hair sweep detail',[(-.18,-.54,4.35),(-.39,-.65,4.12),(-.49,-.71,3.91)],.011,hairlight)
# Two small sage barrettes provide a recognizable, restrained accent.
for z in (3.83,3.95):
    o=rounded('Sage hair clip',(.67,-.57,z),(.20,.05,.055),sage,.025);o.rotation_euler.y=-.35

# Large anime eyes, small nose, blush, and a gentle smile.
for side in (-1,1):
    x=side*.30
    sphere('Eye white',(x,-.657,3.48),(.181,.056,.232),white)
    sphere('Amber iris',(x+.012,-.711,3.466),(.107,.031,.170),iris)
    sphere('Eye pupil',(x+.012,-.739,3.485),(.052,.015,.107),dark)
    sphere('Main eye sparkle',(x-.025,-.755,3.548),(.035,.010,.045),white)
    sphere('Small eye sparkle',(x+.057,-.746,3.402),(.015,.009,.020),white)
    curve('Upper eyelash',[(x-.177,-.68,3.54),(x-.12,-.70,3.67),(x+.03,-.707,3.706),(x+.15,-.68,3.624)],.021,dark)
    outer=x+side*.155
    curve('Outer lash',[(outer,-.68,3.605),(outer+side*.065,-.669,3.66)],.016,dark)
    curve('Eyebrow',[(x-.10,-.626,3.80),(x,-.65,3.826),(x+.10,-.624,3.80)],.014,hair)
    sphere('Blush',(side*.46,-.625,3.23),(.116,.013,.046),blush)
    for k in (-1,0,1):
        curve('Cheek hatch',[(side*.46+k*.035-.008,-.640,3.222),(side*.46+k*.035+.003,-.641,3.244)],.0035,clay)
sphere('Tiny nose',(0,-.718,3.28),(.046,.039,.048),skin)
curve('Smile',[(-.102,-.668,3.134),(0,-.692,3.102),(.102,-.668,3.134)],.013,dark)

# Simple oval plinth gives the render a grounded, collectible feel.
sphere('Oval base',(0,.04,.065),(.93,.58,.075),cream)

bpy.context.preferences.filepaths.save_version=0
scene=bpy.context.scene
scene.render.engine='CYCLES';scene.cycles.samples=64
scene.cycles.use_denoising=True
scene.render.resolution_x=1000;scene.render.resolution_y=1200;scene.render.resolution_percentage=100
scene.render.film_transparent=True
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
scene.world.color=(.65,.65,.65)
scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.72,.69,.64,1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value=.55
scene.view_settings.view_transform='AgX'

def area(name,loc,power,size):
    bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.name=name;o.data.energy=power;o.data.shape='DISK';o.data.size=size
    o.rotation_euler=(Vector((0,0,2.4))-o.location).to_track_quat('-Z','Y').to_euler()
area('Large softbox',(-3,-4,7),450,5)
area('Soft fill',(4,-2,4),180,4)
area('Hair rim',(1,3,6),350,3)
bpy.ops.object.camera_add(location=(0,-13,5.0))
cam=bpy.context.object;cam.name='Hero camera';cam.rotation_euler=(Vector((.08,0,2.32))-cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type='ORTHO';cam.data.ortho_scale=5.12;scene.camera=cam
scene.render.filepath='//../../docs/assets/student-mascot.png'
# Opening the saved file starts in a useful camera view.
for screen in bpy.data.screens:
    for area_ in screen.areas:
        if area_.type=='VIEW_3D':area_.spaces.active.region_3d.view_perspective='CAMERA'
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'art'/'mascot'/'student-mascot.blend'),compress=True)
bpy.ops.render.render(write_still=True)
print('Mascot render complete:',scene.render.filepath)
