import pygame
import numpy as np
import random
import time

# ------------------------
# Initialization
# ------------------------
pygame.init()
SAMPLE_RATE = 44100
pygame.mixer.pre_init(SAMPLE_RATE, size=-16, channels=2, buffer=512)
pygame.mixer.init()
screen = pygame.display.set_mode((1600, 900))
pygame.display.set_caption("Ultimate 50-Synth Playground v6 (Realtime + 50 Engines)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

# ------------------------
# Helper Functions
# ------------------------
def adsr(length, attack, decay, sustain, release):
    n = int(length*SAMPLE_RATE)
    env = np.zeros(n)
    a = int(attack*SAMPLE_RATE)
    d = int(decay*SAMPLE_RATE)
    r = int(release*SAMPLE_RATE)
    s = max(0,n-a-d-r)
    if a>0: env[:a]=np.linspace(0,1,a)
    if d>0: env[a:a+d]=np.linspace(1,sustain,d)
    if s>0: env[a+d:a+d+s]=sustain
    if r>0: env[-r:]=np.linspace(sustain,0,r)
    return env

def make_sound_from_wave(wave):
    stereo = np.column_stack((wave,wave))
    return pygame.sndarray.make_sound(np.int16(np.clip(stereo,-1,1)*32767))

# Simple 1-pole filters and helpers
def one_pole_lowpass(x, cutoff_hz):
    if cutoff_hz <= 0: return x.copy()
    rc = 1.0 / (2*np.pi*cutoff_hz)
    dt = 1.0 / SAMPLE_RATE
    alpha = dt/(rc+dt)
    y = np.zeros_like(x)
    for i,v in enumerate(x):
        y[i] = y[i-1] + alpha*(v - y[i-1]) if i>0 else alpha*v
    return y

def one_pole_highpass(x, cutoff_hz):
    if cutoff_hz <= 0: return x.copy()
    rc = 1.0 / (2*np.pi*cutoff_hz)
    dt = 1.0 / SAMPLE_RATE
    alpha = rc/(rc+dt)
    y = np.zeros_like(x)
    prev_x = 0.0
    for i,v in enumerate(x):
        y[i] = alpha*(y[i-1] + v - prev_x) if i>0 else alpha*(v - 0.0)
        prev_x = v
    return y

def bandpass(x, low, high):
    return one_pole_highpass(one_pole_lowpass(x, high), low)

def soft_clip(x, amt):
    # amt ~ 0..1
    return np.tanh(x*(1+amt*6.0))

def wavefold(x, amt):
    # basic fold using abs/saw to keep stable
    th = 0.6 + 0.35*amt
    y = x.copy()
    over = np.abs(y) > th
    y[over] = th - (np.abs(y[over]) - th)
    y = np.clip(y, -1, 1)
    return y

def chorus(x, depth=0.003, rate=0.8, mix=0.3):
    n = len(x)
    t = np.arange(n)/SAMPLE_RATE
    delay = (depth*(1+np.sin(2*np.pi*rate*t)))*SAMPLE_RATE
    y = x.copy()
    out = np.zeros_like(x)
    for i in range(n):
        j = int(i - delay[i])
        out[i] = x[i]
        if 0 <= j < n:
            out[i] = (1-mix)*x[i] + mix*0.7*x[j]
    return out

def comb_filter(x, delay_ms=20, fb=0.6, mix=0.5):
    d = int(delay_ms * SAMPLE_RATE / 1000.0)
    if d <= 0: return x
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = x[i]
        if i-d >= 0:
            y[i] += fb * y[i-d]
    return np.clip((1-mix)*x + mix*y, -1, 1)

def pwm_square(phase, pw):
    # phase 0..1, pw in (0,1)
    return np.where((phase % 1.0) < pw, 1.0, -1.0)

def tri_from_phase(phase):
    p = (phase % 1.0)
    return 2*np.abs(2*p-1)-1

def saw_from_phase(phase):
    p = (phase % 1.0)
    return 2*p-1

# ------------------------
# Core Sound Generator with 50 Algorithms
# ------------------------
def generate_wave_buffer(frequency, table, alg, duration=0.8,
                         osc=3, detune=0.5, rough=0.1, vol=0.5,
                         attack=0.01, decay=0.05, sustain=0.6, release=0.05,
                         fm_amount=50, distortion=0.5, bitcrush=8, harmonics=5,
                         echo=0.0, pitch=1.0, apply_envelope=False):
    """
    Generates a loopable sustain buffer governed by 'alg' (1..50).
    'table' is the 128-sample editable waveform (used directly or blended).
    """
    frequency *= max(0.01, float(pitch))
    n = int(duration*SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    L = len(table)
    tx = np.arange(L)
    wave = np.zeros(n)

    # Detuned oscillators base (many algs build atop this)
    def table_osc(freq, det=1.0, phase0=0.0):
        ph = (t*freq*det + phase0) % 1.0
        return np.interp(ph*L, tx, table)

    # FM helper
    def fm_osc(freq, ratio=2.0, idx=0.0):
        mod = np.sin(2*np.pi*freq*ratio*t) * idx
        ph = (t*freq + mod) % 1.0
        return np.interp(ph*L, tx, table)

    # AM helper
    def am_apply(car, freq, depth=0.5):
        return car * (1 - depth + depth*np.sin(2*np.pi*freq*t)*0.5 + 0.5)

    # Build by alg
    if alg == 1:  # Pure sine + editor blend
        base = np.sin(2*np.pi*frequency*t)
        wt = table_osc(frequency)
        wave = 0.6*base + 0.4*wt

    elif alg == 2:  # Square (sign of sine) + sub
        base = np.sign(np.sin(2*np.pi*frequency*t))
        sub = np.sign(np.sin(2*np.pi*(frequency/2)*t))
        wave = 0.75*base + 0.25*sub

    elif alg == 3:  # Saw + editor table
        base = saw_from_phase(t*frequency)
        wave = 0.7*base + 0.3*table_osc(frequency)

    elif alg == 4:  # Triangle + soft clip
        base = tri_from_phase(t*frequency)
        wave = soft_clip(base, 0.25)

    elif alg == 5:  # PWM (modulated)
        pw = 0.5 + 0.2*np.sin(2*np.pi*0.4*t)
        wave = pwm_square(t*frequency, np.clip(pw,0.05,0.95))

    elif alg == 6:  # Supersaw (7 voices) + subtle chorus
        dets = np.linspace(-0.015,0.015,7)
        s = 0
        for d in dets:
            s += saw_from_phase(t*frequency*(1+d))
        s /= 7.0
        wave = chorus(s, depth=0.0025, rate=0.6, mix=0.25)

    elif alg == 7:  # Hard-sync like (wrap saw into faster saw)
        master = saw_from_phase(t*frequency)
        slave = saw_from_phase(t*frequency*2.7)
        wave = np.where(master>0, slave, -slave)

    elif alg == 8:  # Phase distortion (Casio-ish)
        ph = (t*frequency) % 1.0
        pd = ph + 0.35*np.sin(2*np.pi*ph)
        wave = saw_from_phase(pd)

    elif alg == 9:  # Sub-osc blend (editor table + strong sub)
        wave = 0.55*table_osc(frequency) + 0.45*np.sin(2*np.pi*(frequency/2)*t)

    elif alg == 10:  # Organ drawbar (additive sines)
        wave = (1.0*np.sin(2*np.pi*frequency*t) +
                0.5*np.sin(2*np.pi*2*frequency*t) +
                0.3*np.sin(2*np.pi*3*frequency*t) +
                0.2*np.sin(2*np.pi*4*frequency*t))

    # FM cluster: 11..20
    elif alg == 11:  # FM bell (ratio 2, index)
        wave = np.sin(2*np.pi*frequency*t + (fm_amount/100.0)*np.sin(2*np.pi*frequency*2*t))

    elif alg == 12:  # FM metallic (ratio 3.7)
        wave = np.sin(2*np.pi*frequency*t + (fm_amount/80.0)*np.sin(2*np.pi*frequency*3.7*t))

    elif alg == 13:  # FM electric piano-ish
        m = (fm_amount/120.0)
        wave = 0.7*np.sin(2*np.pi*frequency*t + m*np.sin(2*np.pi*frequency*2*t)) + 0.3*np.sin(2*np.pi*(frequency/2)*t)

    elif alg == 14:  # FM growl (low ratio)
        wave = np.sin(2*np.pi*frequency*t + (fm_amount/60.0)*np.sin(2*np.pi*frequency*0.5*t))

    elif alg == 15:  # FM formant sweep (moving index)
        idx = (fm_amount/100.0)*(0.5+0.5*np.sin(2*np.pi*0.2*t))
        wave = np.sin(2*np.pi*frequency*t + idx*np.sin(2*np.pi*frequency*2*t))

    elif alg == 16:  # FM + Sub
        wave = 0.75*np.sin(2*np.pi*frequency*t + (fm_amount/95.0)*np.sin(2*np.pi*frequency*1.5*t)) + 0.25*np.sin(2*np.pi*(frequency/2)*t)

    elif alg == 17:  # 2-op FM with table as carrier
        carrier = table_osc(frequency)
        mod = np.sin(2*np.pi*frequency*2*t)*(fm_amount/90.0)
        ph = (t*frequency + mod) % 1.0
        wave = 0.8*np.interp(ph*len(table), np.arange(len(table)), table) + 0.2*carrier

    elif alg == 18:  # Cross-FM (two sines mod each other)
        m1 = (fm_amount/140.0)
        m2 = (fm_amount/220.0)
        a = np.sin(2*np.pi*frequency*t + m1*np.sin(2*np.pi*frequency*2*t))
        b = np.sin(2*np.pi*frequency*1.5*t + m2*np.sin(2*np.pi*frequency*0.75*t))
        wave = 0.6*a + 0.4*b

    elif alg == 19:  # FM noise-edge (fast mod)
        wave = np.sin(2*np.pi*frequency*t + (fm_amount/40.0)*np.sin(2*np.pi*frequency*8*t))

    elif alg == 20:  # FM bell bright (ratio 5)
        wave = np.sin(2*np.pi*frequency*t + (fm_amount/85.0)*np.sin(2*np.pi*frequency*5*t))

    # AM / Ring / Dist cluster: 21..30
    elif alg == 21:  # Ring mod (50%)
        car = table_osc(frequency)
        mod = np.sin(2*np.pi*frequency*2*t)
        wave = car * mod

    elif alg == 22:  # Tremolo AM
        car = table_osc(frequency)
        wave = am_apply(car, frequency*0.5, depth=0.8)

    elif alg == 23:  # Bit-decimator (downsample-ish)
        base = saw_from_phase(t*frequency)
        step = max(1, int( (17-bitcrush) * 2 ))
        hold = base[::step]
        wave = np.repeat(hold, step)[:n]

    elif alg == 24:  # Wavefold lead
        base = 0.7*saw_from_phase(t*frequency) + 0.3*tri_from_phase(t*frequency)
        wave = wavefold(base, 0.8)

    elif alg == 25:  # Hard clipper
        base = 1.6*np.sin(2*np.pi*frequency*t)
        wave = np.clip(base, -0.6, 0.6)/0.6

    elif alg == 26:  # Dist square + sub
        base = np.sign(np.sin(2*np.pi*frequency*t))
        wave = soft_clip(0.8*base + 0.2*np.sin(2*np.pi*(frequency/2)*t), 0.7)

    elif alg == 27:  # AM metallic (high-rate)
        car = tri_from_phase(t*frequency)
        wave = am_apply(car, frequency*10, depth=0.9)

    elif alg == 28:  # Ring + comb
        car = table_osc(frequency)
        mod = np.sin(2*np.pi*frequency*3*t)
        ring = car*mod
        wave = comb_filter(ring, delay_ms=15, fb=0.7, mix=0.5)

    elif alg == 29:  # Formant-ish (bandpass sweeps)
        base = saw_from_phase(t*frequency)
        sweep = 600 + 1200*(0.5+0.5*np.sin(2*np.pi*0.25*t))
        # crude moving BPF: average two bandpasses
        bp1 = bandpass(base, 200, sweep)
        bp2 = bandpass(base, sweep, sweep+1500)
        wave = 0.6*bp1 + 0.4*bp2

    elif alg == 30:  # Gritty ring + bitcrush
        base = table_osc(frequency)
        rm = base*np.sin(2*np.pi*frequency*6*t)
        step = 2**int(bitcrush)
        wave = np.floor(soft_clip(rm,0.6)*step)/step

    # Noise / Texture: 31..40
    elif alg == 31:  # White noise + LPF
        noise = np.random.uniform(-1,1,n)
        wave = one_pole_lowpass(noise, 2000)

    elif alg == 32:  # Pinkish noise (LP multiple stages)
        noise = np.random.uniform(-1,1,n)
        wave = one_pole_lowpass(one_pole_lowpass(noise, 3000), 2000)

    elif alg == 33:  # Noise burst + pitched mixed
        noise = np.random.uniform(-1,1,n)
        tone = np.sin(2*np.pi*frequency*t)
        wave = 0.5*one_pole_lowpass(noise, 1500) + 0.5*tone

    elif alg == 34:  # Granular like: stutter table
        base = table_osc(frequency)
        grain = int(0.02*SAMPLE_RATE)
        wave = np.zeros_like(base)
        i = 0
        while i < n:
            end = min(n, i+grain)
            wave[i:end] = base[i]
            i += grain
        wave = one_pole_lowpass(wave, 4000)

    elif alg == 35:  # Clicky percussive (HP noise)
        noise = np.random.uniform(-1,1,n)
        wave = one_pole_highpass(noise, 4000)

    elif alg == 36:  # Breath pad (LP noise + table)
        noise = np.random.uniform(-0.5,0.5,n)
        pad = one_pole_lowpass(noise, 1200)
        wave = 0.6*pad + 0.4*table_osc(frequency)

    elif alg == 37:  # Vinyl crackle style (sparse impulses)
        wave = np.zeros(n)
        pops = np.random.rand(n) < 0.002
        wave[pops] = np.random.uniform(-1,1,np.sum(pops))
        wave = one_pole_lowpass(wave, 2500)

    elif alg == 38:  # Windy (HP noise with slow AM)
        noise = one_pole_highpass(np.random.uniform(-1,1,n), 1000)
        wave = noise * (0.5 + 0.5*np.sin(2*np.pi*0.2*t))

    elif alg == 39:  # Bit rain (sample+hold noise on grid)
        step = int(SAMPLE_RATE/60)
        base = np.random.uniform(-1,1, n//step + 1)
        wave = np.repeat(base, step)[:n]

    elif alg == 40:  # Dusty pad (LP noise + chorus)
        noise = one_pole_lowpass(np.random.uniform(-1,1,n), 1500)
        wave = chorus(noise, depth=0.004, rate=0.2, mix=0.5)

    # Experimental / Hybrid: 41..50
    elif alg == 41:  # Plucked string (Karplus-Strong-ish via comb)
        exc = np.random.uniform(-1,1, n)
        wave = comb_filter(exc, delay_ms=1000.0*1.0/frequency, fb=0.8, mix=0.9)

    elif alg == 42:  # Vocal-ish (two bandpasses)
        base = saw_from_phase(t*frequency)
        v1 = bandpass(base, 300, 900)
        v2 = bandpass(base, 1400, 3000)
        wave = 0.6*v1 + 0.5*v2

    elif alg == 43:  # Phaser-ish (all-pass feel via mix)
        base = table_osc(frequency)
        a = one_pole_lowpass(base, 800)
        b = one_pole_highpass(base, 800)
        lfo = 0.5+0.5*np.sin(2*np.pi*0.3*t)
        wave = (1-lfo)*a + lfo*b

    elif alg == 44:  # Resonant comb sweep
        base = saw_from_phase(t*frequency)
        sweep_ms = 5 + 15*(0.5+0.5*np.sin(2*np.pi*0.2*t))
        # varying delay: approximate by segmenting
        wave = np.zeros_like(base)
        seg = int(0.01*SAMPLE_RATE)
        for i in range(0, n, seg):
            delay_ms = float(np.mean(sweep_ms[i:i+seg]))
            wave[i:i+seg] = comb_filter(base[i:i+seg], delay_ms=delay_ms, fb=0.7, mix=0.6)[:min(seg, n-i)]

    elif alg == 45:  # Additive 8 partials with falloff
        s = np.zeros(n)
        for h in range(1,9):
            s += (1.0/h) * np.sin(2*np.pi*frequency*h*t)
        wave = s/np.max(np.abs(s))

    elif alg == 46:  # Dual detuned triangles + sub
        a = tri_from_phase(t*frequency*0.995)
        b = tri_from_phase(t*frequency*1.005)
        sub = np.sin(2*np.pi*(frequency/2)*t)
        wave = 0.45*a + 0.45*b + 0.1*sub

    elif alg == 47:  # Hypersaw (11 voices) + HP filter
        dets = np.linspace(-0.02,0.02,11)
        s = 0
        for d in dets:
            s += saw_from_phase(t*frequency*(1+d))
        s /= 11.0
        wave = one_pole_highpass(s, 200)

    elif alg == 48:  # Table-morph (editor <-> sine) via slow LFO
        lfo = 0.5+0.5*np.sin(2*np.pi*0.15*t)
        wave = (1-lfo)*np.sin(2*np.pi*frequency*t) + lfo*table_osc(frequency)

    elif alg == 49:  # Brutal fold + HP + sub-pulse
        base = saw_from_phase(t*frequency)
        folded = wavefold(base, 1.0)
        hp = one_pole_highpass(folded, 500)
        sub = pwm_square(t*(frequency/2), 0.3)
        wave = 0.75*hp + 0.25*sub

    elif alg == 50:  # Ultrakill-ish growler (ring+fold+comb)
        base = table_osc(frequency)
        ring = base * np.sin(2*np.pi*frequency*4*t)
        folded = wavefold(ring, 0.9)
        wave = comb_filter(soft_clip(folded, 0.7), delay_ms=9, fb=0.78, mix=0.6)

    else:
        # Fallback: editor table with detuned osc stack
        s = 0
        for i in range(int(osc)):
            d = 1 + (i-(osc-1)/2)*(detune/50.0)
            s += table_osc(frequency, det=d, phase0=0.0)
        wave = s / max(1, int(osc))

    # Add common detuned layers if requested (for most tonal algs)
    if alg in list(range(1,11)) + list(range(17,21)) + [45,46,47,48,49]:
        if int(osc) > 1:
            stack = 0
            dets = np.linspace(-detune*0.01, detune*0.01, int(osc))
            for d in dets:
                stack += table_osc(frequency*(1+d))
            stack /= int(osc)
            wave = 0.7*wave + 0.3*stack

    # Roughness (add a higher partial)
    if rough != 0:
        wave += rough*np.sin(2*np.pi*frequency*2*t)

    # Harmonics additive (general extra brightness)
    if harmonics > 1:
        for h in range(2, int(harmonics)+1):
            wave += (1.0/h) * np.sin(2*np.pi*frequency*h*t)

    # Normalize pre-FX
    m = np.max(np.abs(wave))
    if m > 0: wave /= m

    # Distortion
    if distortion > 0:
        wave = np.tanh(wave*(1+distortion*5))

    # Bitcrush
    step = 2**int(bitcrush)
    if step > 1:
        wave = np.floor(wave*step)/step

    # Echo (simple feedback delay ~200ms)
    if echo > 0:
        delay = int(0.2 * SAMPLE_RATE)
        echo_wave = np.zeros_like(wave)
        for i in range(delay, len(wave)):
            echo_wave[i] = wave[i] + echo * echo_wave[i-delay]
        wave = (1-echo)*wave + echo*echo_wave

    # Optional ADSR (we skip here; envelope done by channel volume)
    if apply_envelope:
        env = adsr(duration, attack, decay, sustain, release)
        wave *= env

    # Final gain
    wave *= max(0.0001, vol)
    return make_sound_from_wave(wave)

# ------------------------
# Names & Presets
# ------------------------
synth_names = [
    "Sine+WT Blend", "Square+Sub", "Saw+WT", "Triangle SoftClip", "PWM",
    "Supersaw 7V", "HardSync-ish", "PhaseDist", "WT+SubBlend", "Organ Additive",
    "FM Bell", "FM Metallic", "FM EP", "FM Growl", "FM Formant",
    "FM+Sub", "FM Table Carrier", "Cross-FM", "FM Edge", "FM Bell Bright",
    "Ring Mod", "Tremolo AM", "Decimator", "Wavefold Lead", "Hard Clip",
    "DistSquare+Sub", "AM HiRate", "Ring+Comb", "Formant Sweep", "Ring+Crush",
    "White Noise LP", "Pinkish Noise", "Noise+Tone", "Grain Stutter", "HP Clicks",
    "Breath Pad", "Vinyl Pops", "Windy AM", "Bit Rain", "Dusty Chorus",
    "Pluck Comb", "Vocal-ish", "Phaser-ish", "Comb Sweep", "Additive 8",
    "Dual Tri + Sub", "HyperSaw 11V", "Table Morph", "Fold HP + Sub", "Ultrakill Growl"
]

# Create 50 synth entries (engine index == list index +1)
# Each also has its own editable waveform (seeded from common shapes + noise)
preset_waves = [
    np.sin(np.linspace(0,2*np.pi,128)),
    np.sign(np.sin(np.linspace(0,2*np.pi,128))),
    2*np.linspace(0,1,128)-1,
    2*np.abs(2*np.linspace(0,1,128)-1)-1,
    np.random.uniform(-1,1,128)
]

synth_list = []
for i in range(50):
    base = random.choice(preset_waves).copy()
    base += np.random.uniform(-0.2,0.2,128)
    synth_list.append({
        "name": f"{i+1:02d} - {synth_names[i]}",
        "alg": i+1,
        "waveform": np.clip(base, -1, 1),
        # default params (varied a bit so each opens with character)
        "osc": random.randint(1,5),
        "detune": random.uniform(0.3,1.2),
        "rough": random.uniform(0.0,0.4),
        "vol": random.uniform(0.4,0.9),
        "attack": random.uniform(0.001,0.06),
        "decay": random.uniform(0.05,0.25),
        "sustain": random.uniform(0.3,0.85),
        "release": random.uniform(0.05,0.25),
        "fm_amount": random.uniform(30,140),
        "distortion": random.uniform(0.1,0.8),
        "bitcrush": random.randint(3,12),
        "harmonics": random.randint(1,10),
        "echo": random.uniform(0.0,0.45),
        "pitch": 1.0
    })

# ------------------------
# Slider Class
# ------------------------
class Slider:
    def __init__(self,x,y,w,h,min_val,max_val,default,label):
        self.rect = pygame.Rect(x,y,w,h)
        self.min = min_val
        self.max = max_val
        self.value = default
        self.drag=False
        self.label=label
    def draw(self,surf,font):
        pygame.draw.rect(surf,(180,180,180),self.rect)
        hx = self.rect.x + int((self.value-self.min)/(self.max-self.min)*self.rect.w)
        pygame.draw.rect(surf,(255,0,0),(hx-5,self.rect.y,10,self.rect.h))
        surf.blit(font.render(f"{self.label}: {self.value:.3f}",True,(255,255,255)),(self.rect.x,self.rect.y-20))
    def handle_event(self,event):
        if event.type==pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos): self.drag=True
        if event.type==pygame.MOUSEBUTTONUP: self.drag=False
        if event.type==pygame.MOUSEMOTION and self.drag:
            rel_x = max(0,min(self.rect.w,event.pos[0]-self.rect.x))
            self.value=self.min+(rel_x/self.rect.w)*(self.max-self.min)
# ------------------------
# Neon helpers
# ------------------------
def neon_rect(surface, rect, base_color, glow_color, width=2, glow=6):
    """Draw a neon-style rectangle with halo glow."""
    for i in range(glow, 0, -1):
        alpha = int(30 * (i / glow))
        s = pygame.Surface((rect.w + 2*i, rect.h + 2*i), pygame.SRCALPHA)
        pygame.draw.rect(s, (*glow_color, alpha), (0,0,rect.w+2*i, rect.h+2*i), border_radius=rect.height//2)
        surface.blit(s, (rect.x-i, rect.y-i))
    pygame.draw.rect(surface, base_color, rect, width, border_radius=rect.height//2)

def neon_line(surface, p1, p2, base_color, glow_color, width=2, glow=4):
    """Draw neon line with glow."""
    for i in range(glow,0,-1):
        alpha = int(20 * (i/glow))
        s = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.line(s, (*glow_color, alpha), p1, p2, width+i*2)
        surface.blit(s, (0,0))
    pygame.draw.line(surface, base_color, p1, p2, width)

# ------------------------
# Key Mapping (QWERTY row + ASDF row)
# ------------------------
key_freqs = {
    pygame.K_q: 261.63,  # C4
    pygame.K_w: 293.66,  # D4
    pygame.K_e: 329.63,  # E4
    pygame.K_r: 349.23,  # F4
    pygame.K_t: 392.00,  # G4
    pygame.K_y: 440.00,  # A4
    pygame.K_u: 493.88,  # B4
    pygame.K_i: 523.25,  # C5
    pygame.K_o: 587.33,  # D5

    pygame.K_a: 659.25,  # E5
    pygame.K_s: 698.46,  # F5
    pygame.K_d: 783.99,  # G5
    pygame.K_f: 880.00,  # A5
    pygame.K_g: 987.77,  # B5
    pygame.K_h: 1046.50, # C6
    pygame.K_j: 1174.66, # D6
    pygame.K_k: 1318.51, # E6
    pygame.K_l: 1396.91, # F6
}


# ------------------------
# 3D Cube Setup
# ------------------------
vertices = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]
],dtype=float)*100
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(0,4),(1,5),(2,6),(3,7)]

# ------------------------
# Initial State
# ------------------------
current_synth_index = 0
angle = 0
scroll_index = 0
drawing = False
drawing_prev = {"x": None, "y": None}

# Sliders (kept positions; Echo & Pitch included)
sliders = [
    Slider(200,820,400,20,0,0.5,0.1,"Roughness"),
    Slider(200,780,400,20,0,0.2,0.02,"Attack"),
    Slider(200,740,400,20,0,0.5,0.18,"Decay"),
    Slider(200,700,400,20,0,1,0.7,"Sustain"),
    Slider(200,660,400,20,0,0.8,0.15,"Release"),
    Slider(200,620,400,20,0,1,0.6,"Volume"),
    Slider(200,580,400,20,1,5,3,"Oscillators"),
    Slider(200,540,400,20,0,5,0.7,"Detune %"),
    Slider(200,500,400,20,0,200,70,"FM Amount"),
    Slider(200,460,400,20,0,1,0.45,"Distortion"),
    Slider(200,420,400,20,1,16,8,"Bitcrush"),
    Slider(200,380,400,20,1,20,6,"Harmonics"),
    Slider(200,340,400,20,0,1,0.25,"Echo"),
    Slider(200,300,400,20,0.5,2.0,1.0,"Pitch")
]

# ------------------------
# Realtime Sustain System (with crossfade refresh)
# ------------------------
active_notes = {}  # key -> dict(channel, state, time, params, finger, freq)

def params_snapshot():
    s = sliders
    syn = synth_list[current_synth_index]
    return {
        "alg": syn["alg"],
        "waveform": syn["waveform"],
        "osc": int(s[6].value),
        "detune": s[7].value,
        "rough": s[0].value,
        "vol": s[5].value,
        "attack": s[1].value,
        "decay": s[2].value,
        "sustain": s[3].value,
        "release": s[4].value,
        "fm_amount": s[8].value,
        "distortion": s[9].value,
        "bitcrush": int(s[10].value),
        "harmonics": int(s[11].value),
        "echo": s[12].value,
        "pitch": s[13].value
    }

def params_fingerprint(p):
    wf_hash = float(np.sum(np.round(p["waveform"]*128)).item())
    return (
        int(p["alg"]),
        round(p["osc"],3), round(p["detune"],4), round(p["rough"],4),
        round(p["fm_amount"],3), round(p["distortion"],4), int(p["bitcrush"]),
        int(p["harmonics"]), round(p["echo"],4), round(p["pitch"],4), wf_hash
    )

def play_note(key, freq):
    if key in active_notes:
        return
    p = params_snapshot()
    snd = generate_wave_buffer(
        freq, p["waveform"], alg=p["alg"], duration=0.8,
        osc=p["osc"], detune=p["detune"], rough=p["rough"], vol=max(0.0001, p["vol"]),
        attack=p["attack"], decay=p["decay"], sustain=p["sustain"], release=p["release"],
        fm_amount=p["fm_amount"], distortion=p["distortion"], bitcrush=p["bitcrush"],
        harmonics=p["harmonics"], echo=p["echo"], pitch=p["pitch"],
        apply_envelope=False
    )
    ch = snd.play(-1, fade_ms=20)
    ch.set_volume(0.0)
    active_notes[key] = {
        "channel": ch, "state": "attack", "time": pygame.time.get_ticks(),
        "params": p, "finger": params_fingerprint(p), "freq": freq
    }

def release_note(key):
    if key in active_notes and active_notes[key]["state"] != "release":
        active_notes[key]["state"] = "release"
        active_notes[key]["time"] = pygame.time.get_ticks()

def update_envelopes_and_refresh():
    now = pygame.time.get_ticks()
    finished = []
    p_now = params_snapshot()
    finger_now = params_fingerprint(p_now)

    for key, note in list(active_notes.items()):
        ch = note["channel"]
        p = note["params"]
        t = (now - note["time"]) / 1000.0

        # ADSR to channel volume
        if note["state"] == "attack":
            env = t / p["attack"] if p["attack"]>0 and t < p["attack"] else 1.0
            if env >= 1.0:
                note["state"] = "decay"; note["time"] = now
        elif note["state"] == "decay":
            if p["decay"] > 0 and t < p["decay"]:
                env = 1 - (1 - p["sustain"]) * (t / p["decay"])
            else:
                env = p["sustain"]; note["state"] = "sustain"
        elif note["state"] == "sustain":
            env = p["sustain"]
        else:  # release
            if p["release"] > 0 and t < p["release"]:
                start_env = ch.get_volume()
                env = max(0.0, start_env * (1 - t / p["release"]))
            else:
                env = 0.0
                ch.stop()
                finished.append(key)
        ch.set_volume(float(np.clip(env,0,1)))

        # Live refresh during sustain if params changed
        if note["state"] == "sustain":
            if note["finger"] != finger_now:
                prev_env = ch.get_volume()
                snd_new = generate_wave_buffer(
                    note["freq"], p_now["waveform"], alg=p_now["alg"], duration=0.8,
                    osc=p_now["osc"], detune=p_now["detune"], rough=p_now["rough"], vol=max(0.0001, p_now["vol"]),
                    attack=p_now["attack"], decay=p_now["decay"], sustain=p_now["sustain"], release=p_now["release"],
                    fm_amount=p_now["fm_amount"], distortion=p_now["distortion"], bitcrush=p_now["bitcrush"],
                    harmonics=p_now["harmonics"], echo=p_now["echo"], pitch=p_now["pitch"],
                    apply_envelope=False
                )
                ch.fadeout(18)
                ch.play(snd_new, -1, fade_ms=18)
                ch.set_volume(prev_env)
                note["params"] = p_now
                note["finger"] = finger_now

    for key in finished:
        del active_notes[key]

# ------------------------
# Main loop (rendering part)
# ------------------------
running=True
while running:
    screen.fill((10,10,15))  # darker background for neon pop
    mx,my = pygame.mouse.get_pos()
    angle += 0.01

    # ------------------------
    # Left: Synth list
    # ------------------------
    visible_synths = synth_list[scroll_index:scroll_index+10]
    for idx, synth in enumerate(visible_synths):
        sel = (idx+scroll_index==current_synth_index)
        base = (100,100,255) if sel else (70,70,70)
        glow = (150,150,255) if sel else (50,50,50)
        rect = pygame.Rect(10, 10+idx*60, 180,50)
        neon_rect(screen, rect, base, glow, width=2, glow=6)
        screen.blit(font.render(synth["name"],True,(255,255,255)),(rect.x+10,rect.y+12))

    # ------------------------
    # Waveform editor (center-left)
    # ------------------------
    editor_wave = synth_list[current_synth_index]["waveform"]
    editor_rect = pygame.Rect(200,100,400,200)
    neon_rect(screen, editor_rect, (0,255,150), (0,180,120), width=2, glow=8)
    points = [(editor_rect.x + i*editor_rect.w/127,
               editor_rect.y + editor_rect.h/2 - editor_wave[i]*editor_rect.h/2) for i in range(128)]
    pygame.draw.lines(screen,(0,255,180),False,points,2)
    screen.blit(font.render(f"Engine: {synth_list[current_synth_index]['name']}",True,(200,200,255)),(200,75))

    # ------------------------
    # Sliders
    # ------------------------
    for s in sliders:
        s.draw(screen,font)
        # add small neon glow under slider
        

    # ------------------------
    # Event handling
    # ------------------------
    for event in pygame.event.get():
        if event.type==pygame.QUIT: running=False
        for s in sliders: s.handle_event(event)
        if event.type==pygame.MOUSEBUTTONDOWN:
            for idx, synth in enumerate(visible_synths):
                rect = pygame.Rect(10, 10+idx*60, 180,50)
                if rect.collidepoint(event.pos): current_synth_index = idx+scroll_index
            if editor_rect.collidepoint(event.pos):
                drawing = True; drawing_prev["x"]=None; drawing_prev["y"]=None
        if event.type==pygame.MOUSEBUTTONUP:
            drawing=False; drawing_prev["x"]=None; drawing_prev["y"]=None
        if event.type==pygame.MOUSEMOTION and drawing and editor_rect.collidepoint(event.pos):
            x = int((event.pos[0]-editor_rect.x)/editor_rect.w*127)
            y = (editor_rect.y+editor_rect.h/2 - event.pos[1])/(editor_rect.h/2)
            x = np.clip(x,0,127); y = np.clip(y,-1,1)
            if drawing_prev["x"] is not None:
                for xi in range(min(drawing_prev["x"], x), max(drawing_prev["x"], x)+1):
                    if x != drawing_prev["x"]:
                        editor_wave[xi] = drawing_prev["y"] + (y-drawing_prev["y"])*(xi-drawing_prev["x"])/(x-drawing_prev["x"])
            editor_wave[x] = y
            drawing_prev["x"]=x; drawing_prev["y"]=y
        if event.type==pygame.MOUSEBUTTONDOWN:
            if event.button==4: scroll_index = max(0,scroll_index-1)
            if event.button==5: scroll_index = min(len(synth_list)-10,scroll_index+1)
        if event.type==pygame.KEYDOWN and event.key in key_freqs: play_note(event.key, key_freqs[event.key])
        if event.type==pygame.KEYUP and event.key in key_freqs: release_note(event.key)

    # ------------------------
    # Reactive 3D Cube
    # ------------------------
    rotated = vertices.copy()
    Rx = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    Ry = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    rotated = rotated @ Rx @ Ry @ Rz

    if active_notes:
        env_levels = [n["channel"].get_volume() for n in active_notes.values()]
        avg_env = float(np.clip(np.mean(env_levels), 0, 1))
        scale_factor = 1 + 2.0*avg_env
        pulse = np.sin(pygame.time.get_ticks()/120.0) * 40.0 * avg_env
        jitter = np.full(rotated.shape[0], pulse)
        color = (min(255,int(80+175*avg_env)), min(255,int(60+160*sliders[9].value)), min(255,int(240-150*avg_env)))
    else:
        scale_factor = 1.0
        jitter = np.zeros(rotated.shape[0])
        color = (60,60,60)

    rotated += np.column_stack([jitter,jitter,jitter])
    rotated *= scale_factor

    proj = rotated.copy()
    proj[:,0] = proj[:,0]/(proj[:,2]+300)*400 + 1200
    proj[:,1] = proj[:,1]/(proj[:,2]+300)*400 + 450
    for e in edges:
        neon_line(screen, proj[e[0],:2], proj[e[1],:2], color, (120,200,255), width=2, glow=4)

    # ------------------------
    # Reactive bar under editor
    # ------------------------
    bar_color = (0,255,150)
    bar_rect = pygame.Rect(200,320,400,20)
    neon_rect(screen, bar_rect, bar_color, (0,180,120), width=2, glow=6)

    # ------------------------
    # ADSR + live refresh
    # ------------------------
    update_envelopes_and_refresh()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
