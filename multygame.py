import pygame
import socket
import threading
import pickle
import sys
import time

# --- CONFIG ---
WIDTH, HEIGHT = 800, 600
PLAYER_SIZE = 40
BULLET_SIZE = 5
PLAYER_SPEED = 5
BULLET_SPEED = 10
PORT = 5555
BROADCAST_PORT = 5556

# --- GLOBALS ---
players = {}  # {addr: {'x': , 'y': , 'bullets': []}}
client_sockets = []

# --- SERVER FUNCTIONS ---
def handle_client(client_socket, addr):
    global players
    players[addr] = {'x': WIDTH//2, 'y': HEIGHT//2, 'bullets': []}
    while True:
        try:
            data = client_socket.recv(4096)
            if not data:
                break
            state = pickle.loads(data)
            players[addr]['x'] = state['x']
            players[addr]['y'] = state['y']
            players[addr]['bullets'] = state['bullets']
        except:
            break
    del players[addr]
    client_sockets.remove(client_socket)
    client_socket.close()

def server_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('', PORT))
    server.listen()
    print("[SERVER] Hosting game on port", PORT)
    while True:
        client_socket, addr = server.accept()
        client_sockets.append(client_socket)
        threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True).start()

def broadcast_loop():
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while True:
        udp.sendto(b"MULTIPLAYER_HOST", ('<broadcast>', BROADCAST_PORT))
        time.sleep(1)

# --- CLIENT AUTO-DISCOVERY ---
def discover_host(timeout=5):
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp.bind(('', BROADCAST_PORT))
    udp.settimeout(timeout)
    try:
        data, addr = udp.recvfrom(1024)
        if data == b"MULTIPLAYER_HOST":
            print(f"Host found at {addr[0]}")
            return addr[0]
    except socket.timeout:
        print("No host found.")
        return None

# --- GAME LOOP ---
def game_loop(is_host, host_ip=''):
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    
    x, y = WIDTH//2, HEIGHT//2
    bullets = []
    players_state = {}
    
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if not is_host:
        client.connect((host_ip, PORT))
    
    def receive_loop():
        nonlocal players_state
        while True:
            try:
                data = client.recv(4096)
                players_state = pickle.loads(data)
            except:
                break
    
    if not is_host:
        threading.Thread(target=receive_loop, daemon=True).start()
    else:
        players[('host',0)] = {'x': x, 'y': y, 'bullets': []}
    
    run = True
    while run:
        clock.tick(60)
        win.fill((30,30,30))
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: y -= PLAYER_SPEED
        if keys[pygame.K_s]: y += PLAYER_SPEED
        if keys[pygame.K_a]: x -= PLAYER_SPEED
        if keys[pygame.K_d]: x += PLAYER_SPEED
        
        # Shooting
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            dx = mx - x
            dy = my - y
            dist = max(1, (dx**2 + dy**2)**0.5)
            vx, vy = BULLET_SPEED*dx/dist, BULLET_SPEED*dy/dist
            bullets.append({'x': x, 'y': y, 'vx': vx, 'vy': vy})
        
        bullets = [b for b in bullets if 0 <= b['x'] <= WIDTH and 0 <= b['y'] <= HEIGHT]
        
        # Send state
        state = {'x': x, 'y': y, 'bullets': bullets}
        if not is_host:
            try:
                client.send(pickle.dumps(state))
            except:
                pass
        else:
            players[('host',0)]['x'] = x
            players[('host',0)]['y'] = y
            players[('host',0)]['bullets'] = bullets
        
        # Draw players
        current_players = players_state if not is_host else players
        for p in current_players.values():
            pygame.draw.rect(win, (0,255,0), (p['x'], p['y'], PLAYER_SIZE, PLAYER_SIZE))
            for b in p['bullets']:
                pygame.draw.circle(win, (255,0,0), (int(b['x']), int(b['y'])), BULLET_SIZE)
        
        # Draw local player for client
        if not is_host:
            pygame.draw.rect(win, (0,0,255), (x, y, PLAYER_SIZE, PLAYER_SIZE))
        
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
    
    pygame.quit()

# --- MAIN MENU ---
print("1. Host Game")
print("2. Join Game")
choice = input("Choose: ")
if choice == '1':
    threading.Thread(target=server_thread, daemon=True).start()
    threading.Thread(target=broadcast_loop, daemon=True).start()
    game_loop(is_host=True)
elif choice == '2':
    host_ip = discover_host()
    if host_ip:
        game_loop(is_host=False, host_ip=host_ip)
    else:
        print("Cannot find host on LAN.")
else:
    print("Invalid choice")
