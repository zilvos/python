from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

app = Ursina()
window.title = "Motorcycle Mount Demo (Fixed Falling & Sign)"
mouse.locked = True


# --- helper ---
def sign(x):
    return (x > 0) - (x < 0)


# === ENVIRONMENT ===
ground = Entity(
    model='plane',           # plane works best with raycast
    texture='white_cube',
    texture_scale=(50, 50),
    scale=100,
    collider='mesh',         # raycast-compatible collider
    color=color.light_gray,
)
Sky()


# === PLAYER ===
player = FirstPersonController()
player.position = Vec3(0, 2, 0)
player.gravity = 2
player.cursor.visible = False


# === CAMERA LOOK FOR BIKE ===
class FreeLook(Entity):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.sensitivity = 40
        self.pitch = 0
        self.yaw = 0

    def update(self):
        if not self.enabled:
            return
        m = mouse.velocity
        self.yaw += m.x * self.sensitivity
        self.pitch -= m.y * self.sensitivity
        self.pitch = clamp(self.pitch, -85, 85)
        camera.rotation = Vec3(self.pitch, self.yaw, 0)


cam_look = FreeLook()


# === MOTORCYCLE ===
class Motorcycle(Entity):
    def __init__(self, **kwargs):
        super().__init__(
            model='cube',
            color=color.dark_gray,
            scale=(1.2, 0.6, 2.4),
            collider='box',
            **kwargs
        )
        self.front_wheel = Entity(parent=self, model='cylinder', scale=0.25,
                                  rotation=(90, 0, 0), y=-0.35, z=0.9)
        self.rear_wheel = Entity(parent=self, model='cylinder', scale=0.25,
                                 rotation=(90, 0, 0), y=-0.35, z=-0.9)
        self.speed = 0
        self.max_speed = 20
        self.acceleration = 40
        self.brake = 80
        self.friction = 8
        self.turn_speed = 70
        self.y = 0.3

    def update(self):
        if not mount_system.mounted:
            return

        dt = time.dt
        forward = held_keys['w'] - held_keys['s']
        turn = held_keys['d'] - held_keys['a']
        braking = held_keys['space']

        if forward > 0:
            self.speed += self.acceleration * dt
        elif forward < 0:
            self.speed -= self.acceleration * dt
        else:
            self.speed -= self.friction * dt * sign(self.speed)

        if braking:
            self.speed -= self.brake * dt * sign(self.speed)

        self.speed = clamp(self.speed, -self.max_speed * 0.5, self.max_speed)
        self.rotation_y += turn * self.turn_speed * dt * (abs(self.speed) / self.max_speed)
        self.position += self.forward * self.speed * dt
        self.rotation_z = lerp(self.rotation_z, -turn * 10, 4 * dt)

        # spin wheels
        self.front_wheel.rotation_x += self.speed * 40 * dt
        self.rear_wheel.rotation_x += self.speed * 40 * dt


motorcycle = Motorcycle(position=(2, 0.3, -2))


# === MOUNT SYSTEM ===
class MountSystem(Entity):
    def __init__(self, player, bike, cam_look):
        super().__init__()
        self.player = player
        self.bike = bike
        self.cam_look = cam_look
        self.mounted = False

    def input(self, key):
        if key == 'e':
            if not self.mounted and distance(self.player.position, self.bike.position) < 3:
                self.mount()
            elif self.mounted:
                self.dismount()

    def mount(self):
        self.player.enabled = False
        camera.parent = self.bike
        camera.position = Vec3(0, 1, -0.4)
        camera.rotation = Vec3(0, 0, 0)
        self.cam_look.enabled = True
        self.cam_look.pitch = 0
        self.cam_look.yaw = self.bike.rotation_y
        self.mounted = True

    def dismount(self):
        self.player.position = self.bike.position + self.bike.right * 1.5 + Vec3(0, 1, 0)
        self.player.enabled = True
        camera.parent = self.player.camera_pivot
        camera.position = (0, 0, 0)
        camera.rotation = (0, 0, 0)
        self.cam_look.enabled = False
        self.mounted = False


mount_system = MountSystem(player, motorcycle, cam_look)


# === UI ===
speed_text = Text('', position=window.top_left + Vec2(0.02, -0.02))
instructions = Text(
    'E = mount/dismount | W/S throttle | A/D steer | Space brake | Mouse look',
    position=window.bottom_left + Vec2(0.02, 0.04),
    background=True
)


# === UPDATE ===
def update():
    if mount_system.mounted:
        speed_text.text = f"Speed: {int(motorcycle.speed)}"
    else:
        speed_text.text = "On Foot"

    # safety: keep player above ground
    if player.y < 1:
        player.y = 1


app.run()

