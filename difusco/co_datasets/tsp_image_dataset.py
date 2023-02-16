"""TSP (Traveling Salesman Problem) Image Dataset"""

import cv2
import numpy as np
import torch


class TSPImageDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, img_size, point_radius=2, point_color=1, line_thickness=2, line_color=0.5,
               max_points=100):
    self.data_file = data_file
    self.img_size = img_size
    self.point_radius = point_radius
    self.point_color = point_color
    self.line_thickness = line_thickness
    self.line_color = line_color
    self.max_points = max_points

    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def rasterize(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])

    # Rasterize lines
    img = np.zeros((self.img_size, self.img_size))
    for i in range(tour.shape[0] - 1):
      from_idx = tour[i] - 1
      to_idx = tour[i + 1] - 1

      cv2.line(img,
               ((self.img_size - 1) * points[from_idx, ::-1]).astype(int),
               ((self.img_size - 1) * points[to_idx, ::-1]).astype(int),
               color=self.line_color, thickness=self.line_thickness)

    # Rasterize points
    for i in range(points.shape[0]):
      cv2.circle(img, ((self.img_size - 1) * points[i, ::-1]).astype(int),
                 radius=self.point_radius, color=self.point_color, thickness=-1)

    # Rescale image to [-1,1]
    img = 2 * (img - 0.5)

    return img, points, tour

  def __getitem__(self, idx):
    img, points, tour = self.rasterize(idx)

    return img[np.newaxis, :, :], idx
