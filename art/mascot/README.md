# Student Support mascot

An original anime-inspired student companion made in Blender. The warm cream, chestnut, sage, and terracotta palette matches the project website. A notebook and a small wave connect the character to studying and friendly support.

## Files

- `student-mascot.blend`: editable scene, geometry, materials, lighting, and camera.
- `create_mascot.py`: builds the complete scene from basic meshes and curves; no external models or textures.
- `../../docs/assets/student-mascot.png`: transparent render at 1000 × 1200.
- `../../docs/assets/student-mascot.webp`: compressed transparent image used in the hero.

## Render again

Use Blender 4.5 or newer from the repository root:

```sh
blender --background --python art/mascot/create_mascot.py
```

The script also runs with Blender's official `bpy` Python module (created with version 4.5.12, Python 3.11):

```sh
python art/mascot/create_mascot.py
```

Export the website image with Pillow:

```sh
python -c "from PIL import Image; Image.open('docs/assets/student-mascot.png').save('docs/assets/student-mascot.webp', quality=88, method=6)"
```

The website uses a static image with explicit dimensions and descriptive alternative text. It loads no 3D engine or animation scripts.
