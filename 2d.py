import pygame
import sys

pygame.init()
display = (1500, 900)
screen = pygame.display.set_mode(display)
pygame.display.set_caption("Shape")
clock = pygame.time.Clock()
draw = False
size = (15,15)
run = True
shape = []
not_flying = False
# Initialize pygame
pygame.init()
speed=2
# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Basic Character Controller")

#backround setup
backround =pygame.image.load( r"C:\Users\dmill914\Downloads\treebackround.png")
backround_rect = backround.get_rect(topleft=(0, 0))
#beam attack
beam =pygame.image.load( r"C:\Users\dmill914\Downloads\beam.png")





# Clock (for smooth movement)
fly_cooldown = 200
fly_time = 100 
# Player setup
player_model =pygame.image.load( r"C:\Users\dmill914\Downloads\wiz.gif")
player_size = 50
player_color = (0, 200, 50)
player_speed = 5
player = pygame.Rect(WIDTH//2, HEIGHT//2, player_size, player_size)
#firball setup
fireball_size = 20
firball_color = (226,68,43)
fireball = pygame.Rect(WIDTH//2, HEIGHT//2, fireball_size, fireball_size)
fireballspeed = 2
#mana bar setup
mana_width = 20
mana_hieght = 350
mana_bar = pygame.Rect(WIDTH//12, HEIGHT//10, mana_width, mana_hieght)
mana_color = (62, 170, 200)
# Game loop
running = True

while running:
    clock.tick(60)  # limit FPS to 60
    player.y += player_speed
   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            pygame.draw.rect(screen,mana_color,mana_bar)
    mana_hieght
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE] and not_flying == False and mana_hieght <= 350:
        player.y -= player_speed
        fly_time -= 1
        fly_cooldown = 200
        mana_hieght -= 1
        mana_bar = pygame.Rect(WIDTH//12, HEIGHT//10, mana_width, mana_hieght)
        if mana_hieght <=0:
            not_flying = True
            player.y += player_speed
    # Key input
    elif not_flying == True :
        mana_hieght += 5
        mana_bar = pygame.Rect(WIDTH//12, HEIGHT//10, mana_width, mana_hieght)
        if mana_hieght >= 350:
            mana_hieght = 350
            not_flying = False
            
        
    if fireball.left < 0:
        fireball.left = 0
        fireball.x-=fireballspeed
    if fireball.right > WIDTH:
        fireball.right = WIDTH
        fireball.x+=fireballspeed
    if fireball.top < 0:
        fireball.top = 0
    if fireball.bottom > HEIGHT:
        fireball.bottom = HEIGHT

    
        
         
    if keys[pygame.K_e]:
        screen.blit(beam, player) == False
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        player.y -= player_speed
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        player.y += player_speed
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        player.x -= player_speed
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        player.x += player_speed

    # Keep player inside screen
    if player.left < 0:
        player.left = 0
    if player.right > WIDTH:
        player.right = WIDTH
    if player.top < 0:
        player.top = 0
    if player.bottom > 559:
        player.bottom = 559

    screen.fill((30, 30, 30))  # background
    screen.blit(backround, backround_rect)
    screen.blit(player_model, player)
    
    pygame.draw.rect(screen,mana_color,mana_bar)
    pygame.draw.rect(screen,firball_color,fireball)

    backround = pygame.transform.scale(backround, (WIDTH, HEIGHT))
    pygame.display.flip()

    
