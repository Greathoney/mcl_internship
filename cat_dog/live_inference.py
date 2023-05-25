# model_weights.h5 파일 불러오기
from keras.models import load_model

model = load_model('cat_dog/model_keras2.h5')

import pygame
import numpy as np

# pygame init
WIDTH = 1200
HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
FPS = 200

pygame.init()
pygame.display.set_caption("Dog or Cat")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("arial", 20)
clock = pygame.time.Clock()

field = np.full((64, 64, 3), WHITE)

import cv2

# 이미지를 64x64로 변환하고 색은 유지
def preprocessing(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

img = cv2.imread('cat_dog/test_set/cats/cat.4001.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
field = preprocessing(img)

# draw field
def draw_field():
    screen.fill(WHITE)

    for i in range(64):
        for j in range(64):
            pygame.draw.rect(screen, field[i, j], (j*12, i*12, 12, 12))

    # draw grid
    for i in range(65):
        pygame.draw.line(screen, GRAY, (i*12, 0), (i*12, 768))
    for j in range(65):
        pygame.draw.line(screen, GRAY, (0, j*12), (768, j*12))

    
    text = font.render("Dog: " + "{:.2f}".format(float(predict[0])), True, BLACK)
    screen.blit(text, (820, 20))
    text = font.render("Cat: " + "{:.2f}".format(float(predict[1])), True, BLACK)
    screen.blit(text, (820, 40))

    # RED slider
    pygame.draw.line(screen, (255, 0, 0), (850, 400), (1050, 400), 5)
    pygame.draw.circle(screen, current_color, (850 + int(current_color[0]/255*200), 400), 8)
    pygame.draw.circle(screen, BLACK, (850 + int(current_color[0]/255*200), 400), 10, 1)

    # GREEN slider
    pygame.draw.line(screen, (0, 255, 0), (850, 450), (1050, 450), 5)
    pygame.draw.circle(screen, current_color, (850 + int(current_color[1]/255*200), 450), 8)
    pygame.draw.circle(screen, BLACK, (850 + int(current_color[1]/255*200), 450), 10, 1)

    # BLUE slider
    pygame.draw.line(screen, (0, 0, 255), (850, 500), (1050, 500), 5)
    pygame.draw.circle(screen, current_color, (850 + int(current_color[2]/255*200), 500), 8)
    pygame.draw.circle(screen, BLACK, (850 + int(current_color[2]/255*200), 500), 10, 1)

mode = "idle"
current_color = BLACK
predict = model.predict(np.array([field]), verbose=0)[0]

while True:
    draw_field()
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                field = np.full((64, 64, 3), WHITE)

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if event.button == 1:
                if 0 < x < 768 and 0 < y < 768:
                    mode = "drawing"
                elif 830 < x < 1070 and 380 < y < 420:
                    mode = "red"
                elif 830 < x < 1070 and 430 < y < 470:
                    mode = "green"
                elif 830 < x < 1070 and 480 < y < 520:
                    mode = "blue"

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mode = "idle"

        if mode == "drawing":
            x, y = event.pos
            if 0 < x < 768 and 0 < y < 768:
                if np.any(field[y//12, x//12] != current_color):
                    field[y//12, x//12] = current_color
                    predict = model.predict(np.array([field]), verbose=0)[0]
            else:
                mode = "idle"

        if mode == "red":
            x, y = event.pos
            x = np.clip(x, 850, 1050)
            current_color = (int((x - 850)/200*255), current_color[1], current_color[2])


        if mode == "green":
            x, y = event.pos
            x = np.clip(x, 850, 1050)
            current_color = (current_color[0], int((x - 850)/200*255), current_color[2])


        if mode == "blue":
            x, y = event.pos
            x = np.clip(x, 850, 1050)
            current_color = (current_color[0], current_color[1], int((x - 850)/200*255))


    # clock.tick(FPS)