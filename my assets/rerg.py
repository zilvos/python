from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.shaders import lit_with_shadows_shader
import math
import random
from panda3d.core import TransparencyAttrib


app = Ursina()  # <-- must be first


text_apear = 0
held_box = None
hold_point = Entity(parent=camera, position=(0.7, -0.3, 1.5))
DirectionalLight(shadow_map_resolution=(2048, 2048))
player = FirstPersonController(camera=camera)
player.position = (0, 2, 0)
my_sprite_texture = "hat_man.png"
footstep_sound = Audio('footstep.mp3')
sound = Audio(sound_file_name='ceiling-fan-72370.mp3', autoplay=True, loop=True)
sound3 = Audio(sound_file_name='liminal-beat-243458.mp3', autoplay=True, loop=True)
sound.volume = 0.5
sound3.volume = 0.2
camera.parent = player
crouch_height = 1.0
stand_height = 2.0
crouch_speed = 4
player_can_stand = True
player.enabled = False
mouse.locked = True
num_boxes = 20
boxes = []
player.health = 1000000



def distance_xz(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.z - b.z) ** 2)
class Box(Entity):
    def __init__(self, position=(0,1,0), texture='box.jpg'):
        super().__init__(
            model='cube',
            texture=texture,
            collider='box',
            position=position,
            shader = lit_with_shadows_shader,
            scale=1
        )
        self.velocity = Vec3(0,0,0)
        self.mass = 1
        self.gravity = Vec3(0,-9.8,0)
        self.damping = 0.9  # reduce sliding
damage_flash = Entity(
    parent=camera.ui,
    model='quad',
    scale=2,  # full screen
    color=color.rgba(255, 0, 0, 0.3),  # start fully transparent
    double_sided=True,
    unlit=True,
)
hat_man = Entity(
    model='quad',
    texture=my_sprite_texture,
    scale=(4,5),
    position=(0,2.5,100),
    rotation=(0,0,0),
    double_sided=True,
    unlit=True,
)
hat_man.collider = 'box'
eyes = Entity(
    model='quad',
    collider ='sphere',
    texture='eye.png',
    scale=(1,1),
    position=(0,1,10),
    rotation=(0,0,0),
    double_sided=True,
    unlit=True,
)
eyes_list = []
num_eyes = 13

def spawn_eye():
    if len(eyes_list) >= num_eyes:
        return   # stop once we reach the limit

    x = random.uniform(-40, 40)
    z = random.uniform(-40, 40)
    y = 1.5

    eye = Entity(
        model='quad',
        collider='sphere',
        texture='eye.png',
        scale=(1, 1),
        position=(x, y, z),
        rotation=(0, 0, 0),
        double_sided=True,
        unlit=True
    )
    eye.health = .1
    eye.follow_speed = random.uniform(0.01, 0.05)

    eyes_list.append(eye)

    # schedule next spawn in 0.5 seconds
    invoke(spawn_eye, delay=0.0001)

# start spawning
spawn_eye()


# Walls
wall = Entity(model="cube", texture="wall.jpg", collider="box", scale=(10,6,100), texture_scale=(35,1), position=(55,1,0))
wall_2 = Entity(model="cube", texture="wall.jpg", collider="box", scale=(100,6,10), texture_scale=(35,1), position=(0,1,-55),)
wall_3 = Entity(model="cube", texture="wall.jpg", collider="box", scale=(100,6,10), texture_scale=(35,1), position=(0,1,55),)
wall_4 = Entity(model="cube", texture="wall.jpg", collider="box", scale=(10,6,100), texture_scale=(35,1), position=(-55,1,-55))
wall_4 = Entity(model="cube", texture="wall.jpg", collider="box", scale=(10,6,100), texture_scale=(35,1), position=(-55,1,50))
ground = Entity(model='plane', texture='carpet.jpg', scale=(100,1,100), texture_scale=(35,35), collider="box")
ceiling = Entity(model='plane', texture='ceiling.jpg', scale=(100,1,100), texture_scale=(25,25), collider="mesh", position=(0,4,0),double_sided=True,)


# Random pillars
num_pillars = 20
pillars = []
for _ in range(num_pillars):
    x = random.uniform(-55,55)
    z = random.uniform(-55,55)
    y = 0
    height = random.uniform(2,15)
    width = random.uniform(0.5,9)
    length = random.uniform(0.5,30)
    pillar = Entity(model='cube', collider='mesh', texture="wall.jpg", texture_scale=(1,2), position=(x,y+height/2,z), scale=(width,height,length), double_sided=True,shader = lit_with_shadows_shader)
    pillars.append(pillar)
num_pillars = 10
pillars2 = []
for _ in range(num_pillars):
    x = random.uniform(-55,55)
    z = random.uniform(-55,55)
    y = 2
    height = random.uniform(2,15)
    width = random.uniform(0.5,9)
    length = random.uniform(0.5,30)
    Pillar2 = Entity(model='cube', collider='mesh', texture="wall.jpg", texture_scale=(1,2), position=(x,y+height/2,z), scale=(width,height,length), double_sided=True,rotation =0,shader = lit_with_shadows_shader)
    pillars2.append(Pillar2)

def is_looking_at_target(camera_entity, target_entity, threshold=0.9):
    direction_to_target = (target_entity.world_position - camera_entity.world_position).normalized()
    camera_forward = camera_entity.forward.normalized()
    dot_product = dot(direction_to_target, camera_forward)
    return dot_product > threshold

def change_texture():
    if wall.texture.name == 'wall':
        wall.texture = 'wal_2.jpg'
        print('Texture changed to texture2.png')
    else:
        wall.texture = 'wall.jpg'
        print('Texture changed to texture1.png')
collidables = [ground, ceiling, wall, wall_2, wall_3, wall_4,eyes]+pillars + pillars2 + boxes

def move_with_collision(box, dt):
    steps = 4  # smaller steps = more stable collisions
    delta = box.velocity * dt / steps
    for _ in range(steps):
        box.position += delta
        for c in collidables:
            hit = box.intersects(c)
            if hit.hit and hit.normal:
                n = Vec3(hit.normal).normalized()
                # Push box out of collision
                box.position += n * hit.distance
                # Stop velocity along the normal
                vel_along_n = box.velocity.dot(n)
                if vel_along_n < 0:
                    box.velocity -= n * vel_along_n

def physics_update():
    global held_box

    world_colliders = [wall, wall_2, wall_3, wall_4, ground,ceiling] + pillars + pillars2

    for box in boxes:
        if box == held_box:
            # Smoothly move held box to hold point
            target = hold_point.world_position
            box.world_position = lerp(box.world_position, target, 10 * time.dt)
            box.world_rotation = lerp(box.world_rotation, camera.world_rotation, 12 * time.dt)
            box.velocity = (target - box.world_position) / max(time.dt, 0.001)
            continue

        # Apply gravity
        box.velocity += box.gravity * time.dt

        # Apply damping for XZ movement
        box.velocity.x *= box.damping
        box.velocity.z *= box.damping

        # Move box
        box.position += box.velocity * time.dt

        # Collisions with world objects (walls, pillars, ground)
        for collider in world_colliders:
            hit = box.intersects(collider)
            if hit.hit:
                n = Vec3(hit.normal).normalized()

                # If the normal points strongly downward, ignore Y correction (to prevent sinking)
                if n.y < -0.5:
                    n.y = 0
                    n = n.normalized()

                # Push box out of collider
                box.position += n * hit.distance

                # Stop velocity along collision normal
                vel_along_n = box.velocity.dot(n)
                if vel_along_n < 0:
                    box.velocity -= n * vel_along_n

        # Collisions with other boxes (two-way realistic)
        for other in boxes:
            if other == box:
                continue
            hit = box.intersects(other)
            if hit.hit:
                n = Vec3(hit.normal).normalized()

                # Separate both boxes equally
                correction = n * (hit.distance / 2)
                box.position += correction
                other.position -= correction

                # Exchange velocity along the collision normal
                vel_along_n_box = box.velocity.dot(n)
                vel_along_n_other = other.velocity.dot(n)

                # Only resolve if they are moving toward each other
                if vel_along_n_box - vel_along_n_other < 0:
                    box.velocity -= n * vel_along_n_box
                    other.velocity -= n * vel_along_n_other

                    # Swap momentum along the normal
                    box.velocity += n * vel_along_n_other
                    other.velocity += n * vel_along_n_box

        # Keep boxes above ground
        if box.y < 0.5:
            box.y = 0.5
            if box.velocity.y < 0:
                box.velocity.y = 0
def restart_game():
    # Reset player
    player.enable()
    player.position = (0, 1, 0)
    player.health = 100

    # Clear old eyes
    for eye in eyes_list[:]:
        destroy(eye)
        eyes_list.remove(eye)

    # Respawn new eyes
    for _ in range(10):
        x = random.uniform(-40,40)
        z = random.uniform(-40,40)
        eye = Entity(model='quad',
                     collider='sphere',
                     texture='eye.png',
                     scale=(1,1),
                     position=(x,1.5,z),
                     double_sided=True,
                     unlit=True)
        eye.health = 10
        eyes_list.append(eye)

    # Clear UI (remove Game Over text/buttons/overlay)
    destroy(game_over_overlay)
    destroy(game_over_text)
    destroy(restart_button)


def game_over():
    # Disable player and eyes
    player.disable()
    for eye in eyes_list:
        eye.disable()

    # Dark background overlay
    global game_over_overlay, game_over_text, restart_button
    game_over_overlay = Entity(parent=camera.ui, model='quad',
                               color=color.rgba(0,0,0,180), scale=2)

    # Game Over text
    game_over_text = Text("GAME OVER",
                          origin=(0,0),
                          scale=3,
                          color=color.red,
                          y=0.2,
                          parent=camera.ui)

    # Restart button
    restart_button = Button(
        text="Start Game",
        y=-0.1,
        scale=(0.3,0.1),
        color=color.azure,
        parent=camera.ui
    )
    restart_button.on_click = restart_game
eyes.health = .1
status_effects = Entity(parent=camera.ui)
def update():
    hat_man.look_at(camera.world_position)
    hat_man.rotation_z = 0
    camera.rotation_z += math.sin(time.time()*7)*0.05
    camera.rotation_x += math.sin(time.time()*7)*0.03
    camera_forward = camera.forward
    direction_to_object = (hat_man.position - camera.position).normalized()
    dot_product = Vec3.dot(camera_forward,direction_to_object)
    threshold = 0.95
    footstep_sound.volume = 0.3
    player.collider = 'capsule'


    flash_amount = 0.4  # how strong the flash
    flash_speed = 5 
    AGGRO_RANGE = 20  # eyes only start following player within this distance
    ATTACK_RANGE = 2  # eyes damage player within this distance
    
    global eyes

    # Only run this if eyes exists
    for eye in eyes_list[:]:  # copy list so we can remove destroyed eyes
        if eye is None or not eye.enabled:
            continue  # ✅ skip destroyed/disabled eyes

        # Calculate distance to player
        dist_to_player = distance(eye.position, player.position)

        # Only follow player if inside aggro range
        if dist_to_player <= AGGRO_RANGE:
            target_pos = player.position + Vec3(0, 1.5, 0)
            eye.position = lerp(eye.position, target_pos, 0.02)

            # Always look at player
            eye.look_at(player.position)

            # Check collision with boxes
            for box in boxes:
                hit_info = eye.intersects(box)
                if hit_info.hit:
                    eye.health -= 1 * time.dt
                    eye.color = color.red
                    print(f"Eye at {eye.position} health: {eye.health:.1f}")

                    if eye.health <= 0:
                        destroy(eye)
                        eyes_list.remove(eye)  # ✅ remove from list safely
                        print("An eye was destroyed!")
                        break
                else:
                    eye.color = color.white

            # ✅ check distance only if still alive for damage
            if eye in eyes_list and dist_to_player < ATTACK_RANGE:
                player.health -= 10 * time.dt
                print(f"Player health: {player.health:.1f}")

                # make the flash visible
                damage_flash.color = color.rgba(255, 0, 0, flash_amount)



                if player.health <= 0:
                    print("Player is dead!")
                    game_over()

    damage_flash.color = lerp(damage_flash.color, color.rgba(255,0,0,0), flash_speed * time.dt)
    if held_keys['left shift']:
        player.speed = 10
    else:
        player.speed = 5

    if pillar.position > (-55,y+height/2,0) :
        pillar.y +=.1
    if pillar.position < (-55,y+height/2,0) :
        pillar.y +=.1
    if pillar.x >=-55 :
        pillar.y +=.1
    if pillar.x <=-45 :
        pillar.y +=.1        
    if pillar.rotation_y >= 0 :
        pillar.rotation_y +=.1

    # --- Player crouch logic ---
    buffer = 0.05  # small buffer

    # Raycast from feet to full standing height, ignore player itself
    ray_start = player.world_position + Vec3(0, crouch_height, 0)
    check_distance = stand_height - crouch_height + buffer

    ray = raycast(ray_start, Vec3(0,1,0), distance=check_distance, ignore=[player])
    ceiling_blocking = ray.hit

    # Determine target height
    if held_keys['c'] or ceiling_blocking:
        target_height = crouch_height
    else:
        target_height = stand_height

    # Smoothly interpolate height
    player.height = lerp(player.height, target_height, crouch_speed * time.dt)
    player.camera_pivot.y = player.height



    if held_keys['w'] or held_keys['a'] or held_keys['s'] or held_keys['d']:
        camera.rotation_z += math.sin(time.time()*3)*0.05
        camera.rotation_x += math.sin(time.time()*7)*0.05
        camera.rotation_z -= math.sin(time.time()*3)*0.05
        camera.rotation_x -= math.sin(time.time()*7)*0.05
        if not footstep_sound.playing:
            footstep_sound.loop = True
            footstep_sound.play()
    else:
        if footstep_sound.playing:
            footstep_sound.stop()
    if player.y <=-15:
        player.y = 0
        player.x = 0
        player.z = 0
    physics_update()  # run physics

# === MENU SYSTEM ===
menu = Entity(parent=camera.ui)

menu_bg2 = Entity(
    parent=menu,
    model='quad',
    scale=(1, 5),
    psotition =(-2,0),# covers most of the screen
    color=color.rgba(0, 0, 0,0.5),   # semi-transparent black
    z=1,
    x=-1,
    double_sided=True,
    unlit=True,           # ignore lighting
)
menu_bg3 = Entity(
    parent=menu,
    model='quad',
    scale=(1, 5),
    psotition =(-2,0),# covers most of the screen
    color=color.rgba(0, 0, 0,0.4),   # semi-transparent black
    z=1,
    x=-1,
    double_sided=True,
    unlit=True,           # ignore lighting
)

menu_bg4 = Entity(
    parent=menu,
    model='quad',
    scale=(1, 5),
    psotition =(-2,0),# covers most of the screen
    color=color.rgba(0, 0, 0,0.3),   # semi-transparent black
    z=1,
    x=-1,
    double_sided=True,
    unlit=True,           # ignore lighting
)
menu_bg5 = Entity(
    parent=menu,
    model='quad',
    scale=(1, 5),
    psotition =(-2,0),# covers most of the screen
    color=color.rgba(0, 0, 0,0.2),   # semi-transparent black
    z=1,
    x=-1,
    double_sided=True,
    unlit=True,           # ignore lighting
)
menu_bg6 = Entity(
    parent=menu,
    model='quad',
    scale=(1, 5),
    psotition =(-2,0),# covers most of the screen
    color=color.rgba(0, 0, 0,0.1),   # semi-transparent black
    z=1,
    x=-1,
    double_sided=True,
    unlit=True,           # ignore lighting
)

title = Text("My Game", parent=menu, y=0.4,x= -.7, scale=2, origin=(0,0))

def start_game():
    menu.disable()   
    player.enabled = True
    mouse.locked = True   # lock camera control back to game

def quit_game():
    application.quit()

start_button = Button(
    text="Start Game",
    parent=menu,
    y=0.1,
    x= -.7,
    scale=(0.3,0.1),
    color=color.azure
)
start_button.on_click = start_game

quit_button = Button(
    text="Quit",
    parent=menu,
    y=-0.1,
    x= -.7,
    scale=(0.3,0.1),
    color=color.red
)
quit_button.on_click = quit_game


# make sure player is disabled until Start Game
player.enabled = False
mouse.locked = False


def input(key):
    global held_box

    # only allow box controls if player is enabled
    if player.enabled:
        if key == 'left mouse down':
            if held_box is None:
                hit_info = raycast(camera.world_position, camera.forward, distance=5, ignore=[player])
                if hit_info.hit and isinstance(hit_info.entity, Box):
                    held_box = hit_info.entity
            else:
                held_box = None

        if key == 'right mouse down' and held_box:
            held_box.velocity = camera.forward * 25
            held_box = None

    # Escape toggles menu
    if key == 'escape':
        if menu.enabled:
            menu.disable()
            player.enabled = True
            mouse.locked = True
            menu_bg2.x += .1

            menu_bg3.x += .1

            menu_bg4.x += .1

            menu_bg5.x += .1

            menu_bg6.x += .1

        else:
            menu.enable()
            player.enabled = False
            mouse.locked = False
def is_position_free(position, size=1):
    test_box = Entity(position=position, scale=size, collider='box', enabled=False)
    for pillar in pillars:
        if test_box.intersects(pillar).hit:
            destroy(test_box)
            return False
    destroy(test_box)
    return True

def get_highest_y_at_position(x,z,tolerance=0.5):
    highest_y = 0
    for box in boxes:
        if abs(box.x - x)<tolerance and abs(box.z - z)<tolerance:
            top_y = box.y + box.scale_y/2
            if top_y>highest_y:
                highest_y = top_y
    return highest_y

# Spawn boxes
# Optimized box spawning


# Precompute collidables for spawning check
spawn_collidables = [ground, wall, wall_2, wall_3, wall_4] + pillars

def is_position_free_spawn(position, size=1):
    test_box = Entity(position=position, scale=size, collider='box', enabled=False)
    for c in spawn_collidables + boxes:
        if test_box.intersects(c).hit:
            destroy(test_box)
            return False
    destroy(test_box)
    return True

box_height = 1
spawn_attempts = 200  # max attempts per box to find free space

for _ in range(num_boxes):
    for _ in range(spawn_attempts):
        x = random.uniform(-50, 50)
        z = random.uniform(-50, 50)
        # Start on top of ground
        y = 0.5 + box_height / 2
        if is_position_free_spawn((x, y, z), size=1):
            box = Box(position=(x, y, z))
            boxes.append(box)
            break
    else:
        print("Failed to place a box after many attempts")
# Add one specific box
box1 = Box(position=(5,1,5))
boxes.append(box1)


app.run()
