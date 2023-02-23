import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import time
import math

if __name__ == "__main__":
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0,0.0, -5)
    
    quadric = gluNewQuadric()

    i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Draw the sphere at the current position
        posn = math.sin(i / 200)
        quadric = gluNewQuadric()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT) #Clear the screen
        
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(0, posn, 0)
        glColor4f(0.5, 0.2, 0.2, 1) #Put color
        gluSphere(quadric, 1.0, 20, 20)
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)        
        
        i += 1
        