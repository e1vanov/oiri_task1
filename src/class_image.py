import cv2
import matplotlib.pyplot as plt
import numpy as np

class Image:
    
    def __init__(self, path, extension):

        self.path = path
        self.extension = extension
        self.img = cv2.imread(path + '/img.' + extension)
        self.counted_airplanes = -1

    def count_airplanes(self):

        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        subfigs = fig.subfigures(2)

        ax = subfigs[0].subplots(1)
        ax.set_axis_off()
        ax.imshow(self.img[:,:,[2,1,0]])

        ax = subfigs[1].subplots(2, 3)

        for i in range(2):
            for j in range(3):
                ax[i][j].set_axis_off()

        def get_min(img):
            counts, bins = np.histogram(img.ravel(), bins=256, range=[0, 256], density=True)
            bins = bins[:-1]
            def smooth(y, box_pts):
                box = np.ones(box_pts) / box_pts
                y_smooth = np.convolve(y, box, mode='same')
                return y_smooth

            smooth_freq = smooth(counts, 10)
            st = np.argwhere(smooth_freq >= (np.max(smooth_freq) * 0.6))[-1] + 1
            while not (smooth_freq[st - 1] <= smooth_freq[st] <= smooth_freq[st + 1]):
                st += 1

            return st

        par = 3

        dst = np.array(np.where(self.img[:,:,0] > get_min(self.img[:,:,0]), 255, 0), dtype=np.uint8)

        ax[0][0].set_title('Binarization')
        ax[0][0].imshow(dst, cmap='gray')

        dst = cv2.medianBlur(dst, 7)
        ax[0][1].set_title('Median Blur')
        ax[0][1].imshow(dst, cmap='gray')
        dst = cv2.dilate(dst, np.ones((5, 5)), iterations=1)
        ax[0][2].set_title('Dilation')
        ax[0][2].imshow(dst, cmap='gray')

        element1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5 * par, par))
        dst11 = cv2.erode(dst, element1, iterations=1)
        dst12 = cv2.dilate(dst11, element1, iterations=1)
        ax[1][0].set_title('Vertical Opening')
        ax[1][0].imshow(dst12, cmap='gray')

        element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (par, 5 * par))
        dst21 = cv2.erode(dst, element2, iterations=1)
        dst22 = cv2.dilate(dst21, element2, iterations=1)
        ax[1][1].set_title('Horizontal Opening')
        ax[1][1].imshow(dst22, cmap='gray')

        num_labels, labels_im = cv2.connectedComponents(dst12 | dst22)
        ax[1][2].set_title(f'Result: {num_labels - 1} airplanes')
        ax[1][2].imshow(dst12 | dst22, cmap='gray')

        plt.savefig(self.path + '/process.jpg')
        self.counted_airplanes = num_labels - 1
