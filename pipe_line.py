from lib import *
import numpy as np
import pickle

class Pipe_line():
    def __init__(self, classifier_file='svc.pkl', scaler_file='X_scaler.pkl', img_size=(720, 1280)):
        self.zero_heat = np.zeros(img_size, dtype=np.float)
        self.scales = np.linspace(0.8, 2.0, 10)
        self.box_list = []
        self.ystart = 350
        self.ystop = 650

        self.orient = 12
        self.pix_per_cell = 8
        self.cell_per_block = 1

        self.spatial_size = (32, 32)
        self.hist_bins = 100
        self.hist_range = (0, 256)

        with open(classifier_file, 'rb') as f:
            self.svc = pickle.load(f)
        with open(scaler_file, "rb") as f:
            self.X_scaler = pickle.load(f)
    def process(self, image):
        heat = self.zero_heat.copy()
        box_list = []
        for scale in self.scales:
            boxes = find_cars(image,
                              ystart=self.ystart,
                              ystop=self.ystop,
                              scale=scale,
                              svc=self.svc,
                              X_scaler=self.X_scaler,
                              orient=self.orient,
                              pix_per_cell=self.pix_per_cell,
                              cell_per_block=self.cell_per_block,
                              spatial_size=self.spatial_size,
                              hist_bins=self.hist_bins,
                              draw=False)
            if boxes:
                box_list.extend(boxes)
        # Add heat to each box in box list
        heat = add_heat(heat, box_list)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 3)
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        return draw_img

# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# img_file = './test_images/test4.jpg'
# test_img = mpimg.imread(img_file)
# pipe_line = Pipe_line()
# draw_img = pipe_line.process(test_img)
# plt.imshow(draw_img)
# plt.show()