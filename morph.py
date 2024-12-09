import numpy as np
import cv2
from scipy.spatial import Delaunay
import os
import imageio
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

def generate_grid_points(width, height, num_rows, num_cols):
    """
    Generates a grid of points on the image.
    """
    points = []
    x_step = width // (num_cols - 1)
    y_step = height // (num_rows - 1)

    for i in range(num_rows):
        for j in range(num_cols):
            points.append((j * x_step, i * y_step))

    return np.array(points, np.float32)

def triangulate(points):
    """
    Perform Delaunay triangulation on the grid points.
    """
    return Delaunay(points)

def affine_transform(src_tri, dst_tri, shape):
    """
    Applies affine transform on the triangle from src_tri to dst_tri.
    """
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(shape, warp_mat, (shape.shape[1], shape.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    """
    Morph a triangle from img1 and img2 into a new image with alpha blending.
    """
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_image1 = affine_transform(t1_rect, t_rect, img1_rect)
    warp_image2 = affine_transform(t2_rect, t_rect, img2_rect)

    # Ensure warp_image1 and warp_image2 have the same shape
    warp_image1 = cv2.resize(warp_image1, size)
    warp_image2 = cv2.resize(warp_image2, size)

    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask

def morph_images(img1, img2, points1, points2, triangles, alpha):
    """
    Morph the entire image based on alpha blending between img1 and img2.
    """
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    points = (1 - alpha) * points1 + alpha * points2
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

    for tri in triangles:
        x, y, z = tri
        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        morph_triangle(img1, img2, img_morphed, t1, t2, t, alpha)

    return np.uint8(img_morphed)

def load_images(image_paths):
    """
    Load all images from the provided paths and resize them to have the same shape.
    """
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None
        images.append(img)
    
    # Resize all images to the size of the smallest image
    min_height = min(img.shape[0] for img in images)
    min_width = min(img.shape[1] for img in images)

    resized_images = [cv2.resize(img, (min_width, min_height)) for img in images]
    return resized_images

def select_images():
    """
    Opens a file dialog for the user to select multiple images.
    """
    Tk().withdraw()  # Close the root window
    image_paths = askopenfilenames(
        title="Select images to morph",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return list(image_paths)

def main():
    # Manually select images using a file dialog
    image_paths = select_images()
    
    if not image_paths:
        print("No images selected.")
        return

    output_dir = './output'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load all images
    images = load_images(image_paths)
    if images is None:
        return

    # Parameters
    num_frames_per_transition = 20  # Frames for morphing between each pair of images
    num_rows, num_cols = 10, 10  # Grid for triangulation

    # Generate morphed frames
    frame_paths = []
    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]

        height, width = img1.shape[:2]

        # Generate grid points
        points1 = generate_grid_points(width, height, num_rows, num_cols)
        points2 = generate_grid_points(width, height, num_rows, num_cols)

        # Perform Delaunay triangulation on the grid points
        triangles = triangulate(points1).simplices

        # Morph between img1 and img2
        for j in range(num_frames_per_transition):
            alpha = j / (num_frames_per_transition - 1)
            morphed_frame = morph_images(img1, img2, points1, points2, triangles, alpha)
            output_path = os.path.join(output_dir, f'morphed_frame_{i:03d}_{j:03d}.jpg')
            cv2.imwrite(output_path, morphed_frame)
            frame_paths.append(output_path)
            print(f"Saved frame {i+1}-{j+1}/{num_frames_per_transition}")

    # Create GIF
    gif_path = os.path.join(output_dir, 'ANSWER.gif')
    with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    print(f"GIF created: {gif_path}")
    print("Morphing complete. Check the 'output' directory for results.")

if __name__ == "__main__":
    main()
