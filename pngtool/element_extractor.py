import os

import cv2
import numpy as np

from pngtool.img_util import segment_by_hsv, segment_gray, segment_red


class WallRect:
    def __init__(self, rect, dir) -> None:
        self.rect = rect
        self.x, self.y, self.w, self.h = rect
        self.area = self.w * self.h
        self.dir = dir
        self.length = self.w if dir == "h" else self.h


class WallPositioner:
    """
    given a subimage of binary shear wall, find the position of the vertical and horizontal shear walls
    return their bounding rect
    """

    def __init__(self, subimg, origin, char_len) -> None:
        """
        params:
            subimg: the subimage of the shear wall
            origin: the position of the subimage relative to the original whole image
            char_len: characteristic length of the kernel used to erode the image
                      recommended to be the height of the image divided by 25
        """
        self.img = subimg
        self.x, self.y = origin
        self.char_len = char_len
        self.wall_rects = []

    def _find_wall_one_dir(self, orientation):
        """
        find the position of the shear wall in one direction
        params:
            orientation: 'v' for vertical, 'h' for horizontal
        """
        k_size = (self.char_len, 1) if orientation == "v" else (1, self.char_len)
        kernel = np.ones(k_size, np.uint8)
        img = cv2.erode(self.img, kernel, iterations=1)
        cnts = cv2.findContours(img, cv2.MORPH_RECT, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            self.wall_rects.append(WallRect((x + self.x, y + self.y, w, h), orientation))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    def find_wall(self):
        """
        find the position of the shear wall in both directions
        """
        self._find_wall_one_dir("v")
        self._find_wall_one_dir("h")
        return self.wall_rects


class ElementExtractor(object):
    """
    given a stuctGAN generated image, seperate the shear wall and other parts
    the shear walls are in red color
    """

    def __init__(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        self.image = img
        self.hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        self.supported_ele = ["shear_wall", "infill_wall", "window", "door", "background", "beam", "all"]

    def get_mask_by_hsv(self, ele_cate):
        """
        get masks of diffrent elements based on their hsv values
        i.e. set the corresponding position to 1 if the pixel hsv is in range, otherwise 0
        mask has the same size as the image, and has only one channel, consisting of 0 and 1
        so multiple masks can be combined by bitwise or

        Args:
         - ele_cate: the category of the element, should be in [shear_wall, infill_wall, window, door, background]
           which stands for shear wall, infill wall, window, door, background respectively

        Returns:
         the mask of the specific element
        """

        if ele_cate == "shear_wall":  # shear wall red
            # lower1 = [0, 43, 46]
            # upper1 = [10, 255, 255]
            # lower2 = [156, 43, 46]
            # upper2 = [180, 255, 255]
            mask = segment_red(self.image)
        elif ele_cate == "infill_wall":  # infill wall gray
            # lower = [0, 0, 46]
            # upper = [180, 43, 220]
            # mask = cv2.inRange(self.hsv_img, np.array(lower), np.array(upper))
            mask = segment_gray(self.image, 151, 153)
        elif ele_cate == "window":  # window green
            lower = [35, 43, 46]
            upper = [77, 255, 255]
            # mask = cv2.inRange(self.hsv_img, np.array(lower), np.array(upper))
            mask = segment_by_hsv(self.image, lower, upper)
        elif ele_cate == "door":  # door blue
            lower = [115, 100, 100]
            upper = [125, 255, 255]
            # mask = cv2.inRange(self.hsv_img, np.array(lower), np.array(upper))
            mask = segment_by_hsv(self.image, lower, upper)
        elif ele_cate == "background":  # background white
            # lower = [0, 0, 0]
            # upper = [180, 43, 220]
            # mask = cv2.inRange(self.hsv_img, np.array(lower), np.array(upper))
            # mask = segment_by_hsv(self.image, lower, upper)
            mask = segment_gray(self.image, 252, 255)
        elif ele_cate == "beam":
            # black
            mask = segment_gray(self.image, 0, 10)
        elif ele_cate == "all":
            mask = np.any(np.abs(self.image - np.array([255] * 3)) > 10, axis=2).astype(np.uint8) * 255
        else:
            raise ValueError(
                "Unknown object type, type should be in [shear_wall, infill_wall, window, door, background, beam, all]"
            )
        return mask

    def get_all_kind_masks(self):
        for ele in self.supported_ele:
            yield self.get_mask_by_hsv(ele)

    def extract_element(self, tp):
        """
        extract the shear wall or infill wall
        """
        mask = self.get_mask_by_hsv(tp)
        res = cv2.bitwise_and(self.image, self.image, mask=mask)
        binary = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)[1]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        # use opening mophology to remove noise
        kernel = np.ones((5, 5), np.uint8)
        res = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        pix_count = self.count_pixels(mask)
        return res, pix_count

    def draw_rect_bound(self, img, rects, origin=(0, 0), line_width=2):
        """
        draw a rectangle on the image
        """
        x0, y0 = origin
        if not isinstance(rects, list):
            rects = [rects]
        for rect in rects:
            x, y, w, h = rect
            line_color = (0, 255, 0) if w > h else (0, 0, 255)
            if x0 != 0 or y0 != 0:
                x += x0
                y += y0
            cv2.rectangle(img, (x, y), (x + w, y + h), line_color, line_width)
        return img

    def find_bounding(self, img):
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bounding_rects = []
        wall_rects = []
        char_len = int(img.shape[0] / 25)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # bounding_rects.append((x,y,w,h))
            # cv2.rectangle(img_color, (x, y), (x + w, y + h), (36,255,12), 1)
            # seperate vertical and horizontal walls in the sub-img
            sub_img = img[y : y + h, x : x + w]
            # cv2.imshow('img', sub_img)
            # cv2.waitKey(0)
            wall_rects += WallPositioner(sub_img, (x, y), char_len).find_wall()
        self.draw_rect_bound(img_color, list(map(lambda x: x.rect, wall_rects)))
        cv2.imshow("img", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return wall_rects

    def calc_wall_density(self, wall_rects, direction):
        """
        calculate the wall density of along one direction
        """
        struct_L = self.image.shape[0] if direction == "v" else self.image.shape[1]
        total_len = sum(map(lambda x: x.length, wall_rects))
        return total_len / struct_L


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "ImageProcess", "test2.png")
    extractor = ElementExtractor(path)
    sw = extractor.extract()[0]
    wall_rects = extractor.find_bounding(sw)
    wall_dens = extractor.calc_wall_density(wall_rects, "v")
    print(wall_dens)
    print(extractor.calc_alhpa0(wall_dens))
