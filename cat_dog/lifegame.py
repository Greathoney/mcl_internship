import time
import numpy as np
import pygame

# pygame init
WIDTH = 1200
HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
FPS = 200

pygame.init()
pygame.display.set_caption("Life Game")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("arial", 20)

field = np.zeros((100, 100))

# draw field
def draw_field():
    screen.fill(WHITE)

    for i in range(100):
        for j in range(100):
            if field[i, j] == 0:
                pygame.draw.rect(screen, WHITE, (j*8, i*8, 8, 8))
            else:
                pygame.draw.rect(screen, BLACK, (j*8, i*8, 8, 8))

    # # draw grid
    for i in range(101):
        pygame.draw.line(screen, GRAY, (i*8, 0), (i*8, 800))
    for j in range(101):
        pygame.draw.line(screen, GRAY, (0, j*8), (800, j*8))

    # show time step
    text = font.render("Time step: " + str(time_step), True, BLACK)
    screen.blit(text, (820, 20))

    # show mode 
    text = font.render("Mode: " + mode, True, BLACK)
    screen.blit(text, (820, 40))


mode = "edit"  # the mode is edit or run
time_step = 0

while True:
    draw_field()
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if mode == "edit":
                    time_step = 0
                    mode = "run"
                else:
                    mode = "edit"

        if mode == "edit":
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos
                    if 0 < x < 800 and 0 < y < 800:
                        field[y//8, x//8] = 1
                elif event.button == 3:
                    x, y = event.pos
                    if 0 < x < 800 and 0 < y < 800:
                        field[y//8, x//8] = 0

    if mode == "run":
        time_step += 1

        # start = time.time()
        
        field = np.pad(field, 1, 'constant')
        field_new = np.zeros((102, 102))
        for i in range(1, 101):
            for j in range(1, 101):
                count = np.sum(field[i-1:i+2, j-1:j+2]) - field[i, j]
                if field[i, j] == 1:
                    if count == 2 or count == 3:
                        field_new[i, j] = 1
                else:
                    if count == 3:
                        field_new[i, j] = 1
        field = field_new[1:101, 1:101]

        # print(time.time() - start)


    clock = pygame.time.Clock()
    clock.tick(FPS) 