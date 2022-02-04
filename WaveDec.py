import pywt
import numpy as np
from VideoReader import *


def psnr(frame1, frame2):
    mse = np.mean((frame1 - frame2) ** 2)
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def mse(frame1, frame2):
    return np.mean((frame1 - frame2) ** 2)


class WaveDec:
    def __init__(self, read_from, write_to, wavelet, level):
        """
        """
        # TODO grayscale only fn, rgb soon?
        self.read_from = read_from
        self.write_to = write_to
        self.stream = VideoReader()
        self.stream.set_instream(read_from)
        self.wavelet = wavelet
        self.level = level
        ratio = 80 / 100
        self.M = int(ratio * 1280 * 720)  # TODO why number of coeefs is different?
        # self.M = 954415

    def decompose_frame(self, frame):
        coeffs = pywt.wavedec2(frame, self.wavelet, level=self.level)
        return pywt.coeffs_to_array(coeffs)

    def reconstruct_frame(self, compressed_coeffs, compressed_coeffs_slices):
        coeffs = pywt.array_to_coeffs(compressed_coeffs, compressed_coeffs_slices, output_format='wavedec2')
        return pywt.waverec2(coeffs, self.wavelet)

    def independent_frame_compression(self):
        mses = []
        step = 5
        self.stream.set_ofstream("write_mp4_grayscale", "./data/ride_grayscale_indep_comp.mp4")
        for slice in range(0, int(self.stream.amount_of_frames), step):
            print("slice = ", slice)
            frames = self.stream.process_some_mp4frames(slice, slice + step, "return_grayscale")
            for frame in frames:
                composed_coeffs, composed_slices = self.threshold_frame(frame)
                reconstructed_frame = np.uint8(self.reconstruct_frame(composed_coeffs, composed_slices))
                mses.append(mse(frame, reconstructed_frame))
                # print("Initial frame", frame)
                # print("Reconstructed frame", reconstructed_frame)
                cv2.imshow("Initial frame", frame)
                cv2.waitKey(0)
                cv2.imshow("Reconstructed frame", reconstructed_frame)
                cv2.waitKey(0)
                # print("psnr = ", psnr(frame, reconstructed_frame))
                # print("mse  = ", mse(frame, reconstructed_frame))
                cv2.destroyAllWindows()
                self.stream.write_frame_mp4(reconstructed_frame)
        print("sum(mses) / len(mses)", sum(mses) / len(mses))
        self.stream.release_instream()
        self.stream.release_ofstream("write_mp4_grayscale")
        return sum(mses) / len(mses)

    def threshold_frame(self, frame):
        coeff_arr, coeff_slices = self.decompose_frame(frame)
        coeff_arr_threshold = coeff_arr

        # We don't want to compress details so we put them in separate matrix:
        details = np.zeros((len(coeff_arr) - coeff_slices[0][0].stop, coeff_arr.shape[1]))
        index = 0
        for i in range(coeff_slices[0][0].stop, len(coeff_arr)):
            details[index] = coeff_arr[i]
            index += 1

        # Thresholding details. We want to set M details with
        # minimum absolute values to zero get sparser matrix
        val = 10e-5  # Should be estimation of a value st M smallest elements of matrix is smaller then that

        sum_less = 0
        sum_more = 0
        for i in range(details.shape[0]):
            for j in range(details.shape[1]):
                if abs(details[i][j]) < val:
                    sum_less += 1
                    print(details[i][j])
                elif abs(details[i][j]) >= val:
                    sum_more += 1

        print("Total number of details = ", details.shape[1]*details.shape[0])
        print("Number of elements to be compressed = ", sum_less)
        print("Percentage of compression = ", sum_less/details.shape[1]*details.shape[0])

        details[abs(details) < val] = 0

        # We put details and approximation back together
        index = 0
        for i in range(coeff_slices[0][0].stop, len(coeff_arr)):
            coeff_arr_threshold[i] = details[index]
            index += 1

        print("Details arr length = ", coeff_slices[0][0].stop)
        print("Coef arr length = ", len(coeff_arr))
        print("Coef arr shape = ", coeff_arr.shape)
        print("Coef arr total = ", coeff_arr.shape[1] * coeff_arr.shape[0])

        return coeff_arr_threshold, coeff_slices
