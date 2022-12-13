import pygame
import numpy as np


def draw():
    # Initialize Pygame
    pygame.init()

    # Create the canvas with a black background
    canvas_size = (256, 256)
    canvas = pygame.display.set_mode(canvas_size)
    black = (0, 0, 0)
    canvas.fill(black)

    # Create a 2D array to store the pixel data
    pixel_data = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)

    # Handle user input
    radius = 10
    color = (255, 255, 255)  # White
    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Start drawing when the user clicks and drags the mouse
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                # Stop drawing when the user releases the mouse
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                # Get the current position of the mouse
                pos = pygame.mouse.get_pos()

                # Draw a circle at the mouse position
                if drawing:
                    pygame.draw.circle(canvas, color, pos, radius)

        # Update the pixel data array
        pixel_data = pygame.surfarray.pixels2d(canvas)

        # Convert the pixel data to RGB format
        pixel_data = pixel_data.reshape((canvas_size[0], canvas_size[1]))
        pixel_data = pixel_data.astype(np.uint8)

        # Print the pixel data as a 2D array of RGB tuples
        print(pixel_data)

        # Update the screen
        pygame.display.flip()


def nonzero_coordinates(pixel_values):
    coordinates = []
    for row_index, row in enumerate(pixel_values):
        for col_index, value in enumerate(row):
            if value != 0:
                coordinates.append([row_index, col_index])
    return coordinates


if __name__ == "__main__":
    # draw()
    print(nonzero_coordinates([[255, 0, 0], [0, 255, 255], [255, 0, 255]]))
