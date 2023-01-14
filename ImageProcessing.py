import numpy as np


class ImageProcessing:

    @staticmethod
    def spatial_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        out = ImageProcessing.__spatial_convolution(src, kernel)

        ImageProcessing.__clip(out)

        return out

    @staticmethod
    def __spatial_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        dst = ImageProcessing.__image_padding(src, kernel)

        out = ImageProcessing.__corr(src, dst, kernel)

        return out

    @staticmethod
    def frequency_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        out = ImageProcessing.__frequency_convolution(src, kernel)

        ImageProcessing.__clip(out)

        return out

    @staticmethod
    def __frequency_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kernel = kernel[::-1, ::-1].copy()

        len_row, len_col = ImageProcessing.__cal_padding_len(kernel)

        dst = ImageProcessing.__image_padding(src, kernel)

        kernel = ImageProcessing.__circular_shift(dst, kernel)

        src_f = ImageProcessing.__dft(dst)
        kernel_f = ImageProcessing.__dft_2d(kernel)

        out_f = ImageProcessing.__dot(src_f, kernel_f)
        out = ImageProcessing.__idft(out_f).astype(np.int)

        return out[len_row: -len_row, len_col: -len_col]

    @staticmethod
    def image_sharpening(src: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5)) * -1
        kernel[1:-1, 1:-1] = 2
        kernel[2, 2] = 8
        kernel = kernel / np.sum(kernel)

        return ImageProcessing.spatial_convolution(src, kernel)

    @staticmethod
    def edge_detection(src: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3)) * -1
        kernel[1, 1] = 8
        src_gray = ImageProcessing.gray_processing(src)
        return ImageProcessing.spatial_convolution(src_gray, kernel)

    @staticmethod
    def embossing(src: np.ndarray) -> np.ndarray:
        src_gray = ImageProcessing.gray_processing(src)
        kernel = np.array([[-1, -1, 0],
                           [-1, 0, 1],
                           [0, 1, 1]])
        out = ImageProcessing.__spatial_convolution(src_gray, kernel)
        out = ImageProcessing.__clip2(out)

        return out

    @staticmethod
    def gray_processing(src: np.ndarray) -> np.ndarray:
        src_gray = np.dot(src[..., :3], [0.299, 0.5870, 0.1140]).astype(np.uint8)
        return src_gray

    @staticmethod
    def gaussian_blur(src: np.ndarray) -> np.ndarray:
        kernel_row = 5
        kernel_col = 5
        sigma = 1
        kernel = ImageProcessing.gauss_kernel(kernel_row, kernel_col, sigma)
        return ImageProcessing.spatial_convolution(src, kernel)

    @staticmethod
    def motion_blur(src: np.ndarray) -> np.ndarray:
        kernel = np.diag([1, 1, 1, 1, 1])
        kernel = kernel / np.sum(kernel)
        return ImageProcessing.spatial_convolution(src, kernel)

    @staticmethod
    def gauss_kernel(row: int, col: int, sigma: float) -> np.ndarray:
        out = np.zeros((row, col))

        len_row = row // 2
        len_col = col // 2

        for x in range(-len_row, len_row + 1):
            for y in range(-len_col, len_col + 1):
                out[len_row + x, len_col + y] = 1 / (2 * np.pi * sigma ** 2) * np.exp(
                    - (x ** 2 + y ** 2) / (2 * sigma ** 2))

        out = out / np.sum(out)
        return out

    @staticmethod
    def __image_padding(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:

        src_row, src_col, src_height = ImageProcessing.__cal_src_shape(src)
        len_row, len_col = ImageProcessing.__cal_padding_len(kernel)

        dst = np.zeros((src_row + 2 * len_row, src_col + 2 * len_col, src_height))
        if len(src.shape) == 2:
            src = src.reshape((src_row, src_col, src_height))

        dst[len_row:-len_row, len_col:-len_col, :] = src.copy()

        # 四角
        dst[len_row - 1:: -1, len_col - 1::-1, :] = dst[len_row:2 * len_row, len_col:2 * len_col, :]
        dst[len_row - 1:: -1, len_col + src_col:2 * len_col + src_col, :] = dst[
                                                                            len_row:2 * len_row,
                                                                            len_col + src_col - 1:src_col - 1:-1, :]
        dst[len_row + src_row: 2 * len_row + src_row, len_col - 1::-1, :] = dst[
                                                                            len_row + src_row - 1:src_row - 1:-1,
                                                                            len_col:2 * len_col, :]
        dst[len_row + src_row: 2 * len_row + src_row, len_col + src_col:2 * len_col + src_col, :] = dst[
                                                                                                    len_row + src_row - 1: src_row - 1:-1,
                                                                                                    len_col + src_col - 1: src_col - 1:-1,
                                                                                                    :
                                                                                                    ]

        # 上下
        dst[len_row - 1::-1, len_col:-len_col, :] = dst[len_row:2 * len_row, len_col:-len_col, :]
        dst[len_row + src_row: 2 * len_row + src_row, len_col:-len_col, :] = dst[
                                                                             len_row + src_row - 1:src_row - 1:-1,
                                                                             len_col:-len_col, :]

        # 左右
        dst[len_row:-len_row, len_col - 1::-1, :] = dst[len_row:-len_row, len_col:2 * len_col, :]
        dst[len_row:-len_row, len_col + src_col:2 * len_col + src_col, :] = dst[
                                                                            len_row: -len_row,
                                                                            len_col + src_col - 1: src_col - 1: -1, :]
        return dst

    @staticmethod
    def __circular_shift(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """

        :param src:
        :param kernel:
        :return:
        """
        len_row, len_col = ImageProcessing.__cal_padding_len(kernel)

        out = np.zeros((src.shape[0], src.shape[1]))

        out[:len_row + 1, :len_col + 1] = kernel[len_row:, len_col:]
        out[:len_row + 1, -len_col:] = kernel[len_row:, :len_col]
        out[-len_row:, -len_col:] = kernel[:len_row, :len_col]
        out[-len_row:, :len_col + 1] = kernel[:len_row, len_col:]

        return out

    @staticmethod
    def __dft_1d(src_s: np.ndarray) -> np.ndarray:
        src_len = src_s.shape[0]

        x, y = np.mgrid[0:src_len, 0:src_len]
        w = x * y
        base = np.exp(-1j * 2 * np.pi / src_len)
        w_matrix = base ** w
        src_f = np.dot(w_matrix, src_s)
        return src_f

    @staticmethod
    def __dft_2d(src_s: np.ndarray) -> np.ndarray:
        src_row, src_col = src_s.shape

        src_f = np.zeros(shape=(src_row, src_col), dtype=np.complex128)
        src_fx = np.zeros(shape=(src_row, src_col), dtype=np.complex128)

        for i in range(src_row):
            src_fx[i, :] = ImageProcessing.__dft_1d(src_s[i, :])

        for i in range(src_col):
            src_f[:, i] = ImageProcessing.__dft_1d(src_fx[:, i])

        return src_f

    @staticmethod
    def __dft_3d(src: np.ndarray) -> np.ndarray:
        out = np.zeros(src.shape, dtype=np.complex)
        src_row, src_col, src_height = src.shape
        for height in range(src_height):
            out[:, :, height] = ImageProcessing.__dft_2d(src[:, :, height])
        return out

    @staticmethod
    def __dft(src: np.ndarray) -> np.ndarray:
        if len(src.shape) == 2:
            return ImageProcessing.__dft_2d(src)
        else:
            return ImageProcessing.__dft_3d(src)

    @staticmethod
    def __idft_2d(src_f) -> np.ndarray:

        src_s = np.real(np.conjugate(ImageProcessing.__dft_2d(np.conjugate(src_f))) / src_f.size)
        return src_s

    @staticmethod
    def __idft_3d(src: np.ndarray) -> np.ndarray:
        out = np.zeros(src.shape)
        src_row, src_col, src_height = src.shape
        for height in range(src_height):
            out[:, :, height] = ImageProcessing.__idft_2d(src[:, :, height])
        return out

    @staticmethod
    def __idft(src_f: np.ndarray) -> np.ndarray:
        if len(src_f.shape) == 2:
            return ImageProcessing.__idft_2d(src_f)
        else:
            return ImageProcessing.__idft_3d(src_f)

    @staticmethod
    def __dot_2d(src_f: np.ndarray, kernel_f: np.ndarray) -> np.ndarray:
        return src_f * kernel_f

    @staticmethod
    def __dot_3d(src_f: np.ndarray, kernel_f: np.ndarray) -> np.ndarray:
        out = np.zeros(src_f.shape, dtype=np.complex)
        for height in range(src_f.shape[2]):
            out[:, :, height] = ImageProcessing.__dot_2d(src_f[:, :, height], kernel_f)
        return out

    @staticmethod
    def __dot(src_f: np.ndarray, kernel_f: np.ndarray) -> np.ndarray:
        if len(src_f.shape) == 2:
            return ImageProcessing.__dot_2d(src_f, kernel_f)
        else:
            return ImageProcessing.__dot_3d(src_f, kernel_f)

    @staticmethod
    def __corr(src: np.ndarray, dst: np.ndarray, kernel: np.ndarray) -> np.ndarray:

        src_row, src_col, src_height = ImageProcessing.__cal_src_shape(src)
        len_row, len_col = ImageProcessing.__cal_padding_len(kernel)

        out = np.zeros(src.shape, dtype=np.int)
        if len(out.shape) == 2:
            out = out.reshape((src_row, src_col, src_height))
        # 卷积
        for height in range(src_height):
            for row in range(len_row, len_row + src_row):
                for col in range(len_col, len_col + src_col):
                    out[row - len_row, col - len_col, height] = np.round(np.sum(
                        dst[row - len_row:row + len_row + 1, col - len_col:col + len_col + 1, height] * kernel
                    ))

        return out

    @staticmethod
    def __cal_src_shape(src: np.ndarray) -> (int, int, int):
        if len(src.shape) == 2:
            src_height = 1
            src_row, src_col = src.shape
        else:
            src_row, src_col, src_height = src.shape
        return src_row, src_col, src_height

    @staticmethod
    def __cal_padding_len(kernel: np.ndarray) -> (int, int):
        kernel_row, kernel_col = kernel.shape
        len_row = kernel_row // 2
        len_col = kernel_col // 2
        return len_row, len_col

    @staticmethod
    def __clip(src: np.ndarray) -> None:
        min_value = 0
        max_value = 255

        src[src < min_value] = min_value
        src[src > max_value] = max_value

    @staticmethod
    def __clip2(src: np.ndarray) -> np.ndarray:
        return (src - src.min()) / (src.max() - src.min())
