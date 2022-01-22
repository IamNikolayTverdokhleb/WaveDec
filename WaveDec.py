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
    def __init__(self, read_from, write_to, wavelet):
        """
        """
        # TODO grayscale only fn, rgb soon?
        self.read_from = read_from
        self.write_to = write_to
        self.stream = VideoReader()
        self.stream.set_instream(read_from)
        self.wavelet = wavelet
        ratio = 80 / 100
        self.M = int(ratio * 1280 * 720) # TODO why number of coeefs is different?
        #self.M = 954415

    def decompose_frame(self, frame):
        coeffs = pywt.wavedec2(frame, self.wavelet)
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
            frames = self.stream.process_some_mp4frames(slice, slice+step, "return_grayscale")
            for frame in frames:
                composed_coeffs, composed_slices = self.compress_frame(frame)
                reconstructed_frame = np.uint8(self.reconstruct_frame(composed_coeffs, composed_slices))
                mses.append(mse(frame, reconstructed_frame))
                # print("Initial frame", frame)
                # print("Reconstructed frame", reconstructed_frame)
                # cv2.imshow("Initial frame", frame)
                # cv2.waitKey(0)
                # cv2.imshow("Reconstructed frame", reconstructed_frame)
                # cv2.waitKey(0)
                # print("psnr = ", psnr(frame, reconstructed_frame))
                # print("mse  = ", mse(frame, reconstructed_frame))
                # cv2.destroyAllWindows()
                self.stream.write_frame_mp4(reconstructed_frame)
        print("sum(mses) / len(mses)", sum(mses) / len(mses))
        self.stream.release_instream()
        self.stream.release_ofstream("write_mp4_grayscale")
        return sum(mses) / len(mses)

    def compress_frame(self, frame):
        coeff_arr, coeff_slices = self.decompose_frame(frame)
        # Thresholding (lenght(f) - M) lowest DETAIL coefficients
        m_biggest = np.sort(coeff_arr[coeff_slices[0][0].stop:])[0:self.M]

        coeff_arr_compressed = coeff_arr  # to store compressed values
        # print("Coef arr length = ", len(coeff_arr))
        # print("Coef arr shape = ", coeff_arr.shape)
        # print("Coef arr total = ", coeff_arr.shape[1]*coeff_arr.shape[0])
        # print("M = ", self.M)
        for i in range(coeff_slices[0][0].stop, len(coeff_arr)):
            if not coeff_arr[i] in m_biggest:  # if that value is not among M biggest we set it to zero
                coeff_arr_compressed[i] = 0
                #print("Why we here???")

        return (coeff_arr_compressed, coeff_slices)
