from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader
import math
import os
import random


app = Ursina()
text_apear = 0
held_box = None
DirectionalLight(shadow_map_resolution=(2048, 2048))
player = FirstPersonController(camera=camera)
player.position = (0, 5, 0)
my_sprite_texture = "hat_man.png"
footstep_sound = Audio('footstep.mp3')
sound = Audio(sound_file_name='ceiling-fan-72370.mp3', autoplay=True, loop=True)
sound3 = Audio(sound_file_name='liminal-beat-243458.mp3', autoplay=True, loop=True)
sound.volume = 0.5
sound3.volume = 0.2
camera.parent = player


class Box(Entity):
    def __init__(self, position=(0, 1, 0), texture='box.jpg'):
        super().__init__(
            model='cube',
            texture=texture,
            collider='box',
            position=position,
            scale=1
        )
        self.original_parent = self.parent  # Store original parent for placing


hat_man = Entity(
    model='quad',
    texture=my_sprite_texture,
    scale=(4, 5),
    position=(0, 2.5, 1),
    rotation=(0, 0, 0),
    double_sided=True,
    unlit=True,

)

hat_man.collider = 'box'


wall = Entity(
    model="cube",
    texture="wall.jpg",
    collider="box",
    scale=(10, 6, 100),
    texture_scale=(35, 1),
    position=(55, 1, 0))
wall_2 = Entity(
    model="cube",
    texture="wall.jpg",
    collider="box",
    scale=(100, 6, 10),
    texture_scale=(35, 1),
    shader=lit_with_shadows_shader,
    position=(0, 1, -55))
wall_3 = Entity(
    model="cube",
    texture="wall.jpg",
    collider="box",
    scale=(100, 6, 10),
    texture_scale=(35, 1),
    position=(0, 1, 55))
wall_4 = Entity(
    model="cube",
    texture="wall.jpg",
    collider="box",
    scale=(10, 6, 100),
    texture_scale=(35, 1),
    position=(-55, 1, 0))

ground = Entity(
    plane=Entity(model='plane', texture='carpet.jpg', scale=(100, 1, 100), texture_scale=(35, 35), collider="mesh",),

)
ceiling = Entity(
    plane=Entity(model='plane', texture='ceiling.jpg', scale=(100, 1, 100), texture_scale=(25, 25), collider="mesh", position=(0, 4, 0), rotation_x=180, unlit=True),

)

num_pillars = 500
pillars = []

for _ in range(num_pillars):

    x = random.uniform(-55, 55)
    z = random.uniform(-55, 55)
    y = 0

    height = random.uniform(2, 15)
    width = random.uniform(0.5, 3)
    pillar = Entity(
        model='cube',
        collider='box',
        texture="wall.jpg",
        texture_scale=(1, 2),
        position=(x, y + height / 2, z),
        scale=(width, height, width),
        double_sided=True,
    )
    pillars.append(pillar)

    if pillar.position == player.position:
        pillar.position = (x, y + height / 2, z)


def is_looking_at_target(camera_entity, target_entity, threshold=0.9):
    # Get direction from camera to target
    direction_to_target = (target_entity.world_position - camera_entity.world_position).normalized()
    # Get camera's forward vector
    camera_forward = camera_entity.forward.normalized()
    # Calculate dot product
    dot_product = dot(direction_to_target, camera_forward)
    return dot_product > threshold


# --- Shooting mechanic additions start here ---

bullets = []


class Bullet(Entity):
    def __init__(self, position, direction, **kwargs):
        super().__init__(
            model='sphere',
            color=color.yellow,
            scale=0.2,
            position=position,
            collider='sphere',
            **kwargs
        )
        self.direction = direction.normalized()
        self.speed = 40

    def update(self):
        self.position += self.direction * self.speed * time.dt

        # Check collision with hat_man
        if self.intersects(hat_man).hit:
            print("Hat_man hit!")
            destroy(hat_man)
            destroy(self)
            bullets.remove(self)
            return

        # Destroy bullet if too far away
        if distance(self.position, player.position) > 100:
            destroy(self)
            if self in bullets:
                bullets.remove(self)


def update():
    hat_man.look_at(camera.world_position)
    hat_man.rotation_z = 0
    camera.rotation_z += math.sin(time.time() * 7) * 0.05
    camera.rotation_x += math.sin(time.time() * 7) * 0.03
    camera_forward = camera.forward

    direction_to_object = (hat_man.position - camera.position).normalized()

    dot_product = Vec3.dot(camera_forward, direction_to_object)

    threshold = 0.95
    footstep_sound.volume = 0.3

    if player.y <= -20:
        player.position == (0, 0, 0)
    if held_keys['w'] or held_keys['a'] or held_keys['s'] or held_keys['d']:
        camera.rotation_z += math.sin(time.time() * 3) * 0.05
        camera.rotation_x += math.sin(time.time() * 7) * 0.05
        camera.rotation_z -= math.sin(time.time() * 3) * 0.05
        camera.rotation_x -= math.sin(time.time() * 7) * 0.05
        if not footstep_sound.playing:
            footstep_sound.loop = True
            footstep_sound.play()
    else:
        if footstep_sound.playing:
            footstep_sound.stop()

    if hat_man.intersects(pillar):
        # Code to execute when player and enemy collide
        print("Collision detected!")
        hat_man.position = hat_man.position
    if dot_product > threshold:
        print("Looking at the target object!")
        hat_man.z = hat_man.z
        hat_man.x = hat_man.x

    else:
        hat_man.x += 0
        hat_man.z += 0

    # Update bullets
    for bullet in bullets:
        bullet.update()


def input(key):
    global held_box

    if key == 'left mouse down':
        if held_box is None:
            hit_info = raycast(camera.world_position, camera.forward, distance=5, ignore=[player])
            if hit_info.hit and isinstance(hit_info.entity, Box):
                held_box = hit_info.entity
                held_box.parent = camera
                held_box.position = (0.7, 0.01, 0.7)  # Adjust relative position
                held_box.rotation = (0, 0, 0)  # Reset rotation
        else:
            # Place down the box
            held_box.parent = ground  # Unparent from camera
            place_pos = player.position + camera.forward * 2
            # Place box stacking on others
            highest_y = get_highest_y_at_position(place_pos.x, place_pos.z)
            held_box.position = Vec3(place_pos.x, highest_y + held_box.scale_y / 2, place_pos.z)
            held_box = None


def is_position_free(position, size=1):
    test_box = Entity(position=position, scale=size, collider='box', enabled=False)
    for pillar in pillars:
        if test_box.intersects(pillar).hit:
            destroy(test_box)
            return False
    destroy(test_box)
    return True


def get_highest_y_at_position(x, z, tolerance=0.5):
    """Return highest y among boxes near (x, z), or 0 if none"""
    highest_y = 0
    for box in boxes:
        # Check if box is close enough horizontally
        if abs(box.x - x) < tolerance and abs(box.z - z) < tolerance:
            top_y = box.y + box.scale_y / 2  # top surface of the box
            if top_y > highest_y:
                highest_y = top_y
    return highest_y


# Spawn stacked boxes in free spots avoiding pillars and stacking properly
num_boxes = 20
boxes = []

for _ in range(num_boxes):
    for _ in range(100):  # max tries to find free spot
        x = random.uniform(-50, 50)
        z = random.uniform(-50, 50)

        # Get height to stack on
        highest_y = get_highest_y_at_position(x, z)
        box_height = 1
        y = highest_y + box_height / 2

        # Check if this position is free (including stacked y)
        if is_position_free((x, y, z)):
            box = Box(position=(x, y, z))
            boxes.append(box)
            break
    else:
        print("Failed to find free spot for a box")


box1 = Box(position=(5, 1, 5))

player = FirstPersonController()

app.run()

