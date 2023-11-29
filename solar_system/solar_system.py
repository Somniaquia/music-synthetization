import pygame
import math
from win32api import GetSystemMetrics

class CelestialObject:
    def __init__(self, image, radius, center, orbit, speed, color, name, explanation):
        self.image = image
        self.radius = radius
        self.center = center
        self.orbit = orbit
        self.speed = speed
        self.name = name
        self.explanation = explanation
        self.angle = 0
        self.color = color

    def draw(self, camera_position, zoom):
        a, b = [x * zoom for x in self.orbit]
        center_x, center_y = [(x - y) * zoom for x,
                              y in zip(self.center, camera_position)]

        orbit_x = a * math.cos(self.angle)
        orbit_y = b * math.sin(self.angle)

        self.angle += self.speed

        screen_x = center_x + orbit_x
        screen_y = center_y + orbit_y

        ellipse_rect = pygame.Rect(center_x - a, center_y - b, 2*a, 2*b)
        pygame.draw.ellipse(screen, (255, 255, 255), ellipse_rect, 1)

        sprite = self.image
        scaled_sprite = pygame.transform.scale(
            sprite, (int(self.radius * 2 * zoom), int(self.radius * 2 * zoom)))

        screen.blit(scaled_sprite, (int(screen_x - self.radius * zoom), int(screen_y - self.radius * zoom)))
        
    def focus_camera(self):
        pass

pygame.init()

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)

camera_position = (width / 2, height / 2)
zoom = 1
zoom_factor = 1.005
speed = 1.5

smooth_camera_position = camera_position
smooth_zoom = zoom

screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

celestial_objects = []

celestial_objects.append(CelestialObject(pygame.image.load("solar_system/sun.png"), 109, (width / 2, height / 2), (0, 0), 0, (255, 255, 0), "Sun", "The Sun is the center of the Solar System. It is a star composed of hydrogen and helium."))
celestial_objects.append(CelestialObject(pygame.image.load("solar_system/mercury.png"), 1, (width / 2, height / 2), (58+109, 58+109), 0.004, (169, 169, 169), "Mercury", "Mercury is the smallest planet in the Solar System and the closest to the Sun."))
celestial_objects.append(CelestialObject(pygame.image.load("solar_system/venus.png"), 2, (width / 2, height / 2), (108+109, 108+109), 0.0015, (255, 223, 0), "Venus", "Venus is similar in size to Earth but is shrouded in thick clouds of sulfuric acid."))
celestial_objects.append(CelestialObject(pygame.image.load("solar_system/earth.png"), 2, (width / 2, height / 2), (150+109, 150+109), 0.001, (0, 0, 255), "Earth", "Earth is our home planet, with a large amount of water and a breathable atmosphere."))
celestial_objects.append(CelestialObject(pygame.image.load("solar_system/mars.png"), 1, (width / 2, height / 2), (228+109, 228+109), 0.0008, (255, 0, 0), "Mars", "Mars, known as the Red Planet, has a thin atmosphere and is home to the largest volcano and canyon in the Solar System."))
celestial_objects.append(CelestialObject(pygame.image.load("solar_system/jupiter.png"), 11, (width / 2, height / 2), (778+109, 778+109), 0.0004, (255, 165, 0), "Jupiter", "Jupiter is the largest planet in our Solar System and is known for its Great Red Spot."))
celestial_objects.append(CelestialObject(pygame.image.load("solar_system/saturn.png"), 9, (width / 2, height / 2), (1433+109, 1433+109), 0.0003, (255, 215, 0), "Saturn", "Saturn is famous for its prominent ring system and is composed mostly of hydrogen and helium."))
celestial_objects.append(CelestialObject(pygame.image.load("solar_system/uranus.png"), 4, (width / 2, height / 2), (2872+109, 2872+109), 0.0002, (173, 216, 230), "Uranus", "Uranus has a unique blue-green color due to methane in its atmosphere and rotates on its side."))
celestial_objects.append(CelestialObject(pygame.image.load("solar_system/neptune.png"), 4, (width / 2, height / 2), (4495+109, 4495+109), 0.0001, (0, 0, 205), "Neptune", "Neptune is known for its strong winds and was the first planet discovered through mathematical prediction."))

focused_planet = None

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for planet in celestial_objects:
                orbit_x = planet.orbit[0] * math.cos(planet.angle) * smooth_zoom
                orbit_y = planet.orbit[1] * math.sin(planet.angle) * smooth_zoom

                screen_x = (planet.center[0] + orbit_x - smooth_camera_position[0]) * smooth_zoom
                screen_y = (planet.center[1] + orbit_y - smooth_camera_position[1]) * smooth_zoom

                if (mouse_x - screen_x)**2 + (mouse_y - screen_y)**2 <= (planet.radius * smooth_zoom)**2:
                    focused_planet = planet
                    break

    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_UP]:
        camera_position = (camera_position[0], camera_position[1] - speed / smooth_zoom)
        focused_planet = None
    if keys[pygame.K_DOWN]:
        camera_position = (camera_position[0], camera_position[1] + speed / smooth_zoom)
        focused_planet = None
    if keys[pygame.K_LEFT]:
        camera_position = (camera_position[0] - speed / smooth_zoom, camera_position[1])
        focused_planet = None
    if keys[pygame.K_RIGHT]:
        camera_position = (camera_position[0] + speed / smooth_zoom, camera_position[1])
        focused_planet = None
    
    if keys[pygame.K_EQUALS]:
        camera_position = (
            camera_position[0] - (width / 2 - camera_position[0]) / (zoom_factor - 1) / smooth_zoom,
            camera_position[1] - (height / 2 - camera_position[1]) / (zoom_factor - 1) / smooth_zoom
        )
        zoom *= zoom_factor

    if keys[pygame.K_MINUS]:
        camera_position = (
            camera_position[0] + (width / 2 - camera_position[0]) / (1 - 1 / zoom_factor) / smooth_zoom,
            camera_position[1] + (height / 2 - camera_position[1]) / (1 - 1 / zoom_factor) / smooth_zoom
        )
        zoom /= zoom_factor

    smooth_zoom += (zoom - smooth_zoom) * 0.01
    smooth_camera_position = (camera_position[0] + (camera_position[0] - smooth_camera_position[0]) * 0.01, camera_position[1] + (camera_position[1] - smooth_camera_position[1]) * 0.01)

    if keys[pygame.K_SPACE]:
        focused_planet = None

    if focused_planet:
        camera_position = (
            focused_planet.center[0] - width / 2 / smooth_zoom, 
            focused_planet.center[1] - height / 2 / smooth_zoom
        )

    screen.fill((0, 0, 0))
    for planet in celestial_objects:
        planet.draw(smooth_camera_position, smooth_zoom)
    pygame.display.flip()