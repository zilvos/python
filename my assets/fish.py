from ursina import *


app = Ursina()

EditorCamera()
fish = Entity(model = 'red_sea_bream_anil.obj',position = (0,0,0),texture ='red_sea_bream_anil.jpg', double_sided = True )
fish.texture_scale = (1,1) 

fish.texture_offset = (0.009, 0.01) 
def update():
    fish.rotation_y += 1
    if held_keys["space"]:
        fish.rotation_y += 10
    else:
        fish.rotation_y += 1
    if held_keys["1"]:
        
        fish.model = 'red_sea_bream_anil.obj'
    if held_keys["2"]:
        fish.model= 'cube'


app.run()
