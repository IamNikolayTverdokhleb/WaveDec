import pywt
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from VideoReader import *


def psnr(frame1, frame2):
    """
    This function calculates PNSR of two frames
    """
    mse = np.mean((frame1 - frame2) ** 2)
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def mse(frame1, frame2):
    """
    This function calculates MSE of two frames
    """
    return np.mean((frame1 - frame2) ** 2)


class WaveDec:
    def __init__(self, read_from, write_to, wavelet, level):
        """
        """
        # TODO grayscale only fn, rgb soon?
        self.read_from = read_from
        self.write_to = write_to
        self.stream = VideoReader()
        self.stream.set_instream_mp4(read_from)
        self.wavelet = wavelet
        self.level = level
        self.ratio = 99 / 100  # Compression ratio

    def decompose_frame(self, frame):
        """
        This method performs wavelet decomposition of two frames
        """
        coeffs = pywt.wavedec2(frame, self.wavelet, level=self.level)
        return pywt.coeffs_to_array(coeffs)

    def reconstruct_frame(self, compressed_coeffs, compressed_coeffs_slices):
        """
        This method reconstructs frame from wavelet decomposition
        """
        coeffs = pywt.array_to_coeffs(compressed_coeffs, compressed_coeffs_slices, output_format='wavedec2')
        return pywt.waverec2(coeffs, self.wavelet)

    def compress_frame(self, frame):
        """
        This method does decomposition and thresholding of initial frame
        and then puts coefficients into sparse matrix to profit sparsity
        of threshold matrix
        """
        coeff_arr, coeff_slices = self.decompose_frame(frame)
        thrhld_coeff_arr, thrhld_coeff_slices = self.threshold_frame(coeff_arr, coeff_slices)
        coo_compres_coeff_arr = self.from_dense_to_sparce(thrhld_coeff_arr)
        return coo_compres_coeff_arr, thrhld_coeff_slices

    def threshold_frame(self, coeff_arr, coeff_slices):
        """
        This method does thresholding of the detail coefficients
        """
        coeff_arr_threshold = coeff_arr

        # We don't want to threshold details so we put them in separate matrix:
        details = np.zeros((len(coeff_arr) - coeff_slices[0][0].stop, coeff_arr.shape[1]))
        index = 0
        for i in range(coeff_slices[0][0].stop, len(coeff_arr)):
            details[index] = coeff_arr[i]
            index += 1

        # Thresholding details. We want to set M details with
        # minimum absolute values to zero get sparser matrix

        # Estimating value which is bigger then M smallest elements of details matrix
        M = int(self.ratio * details.shape[0] * details.shape[1])  # How many coeffs to threshold
        details_flat_sorted = sorted(details.flatten(), key=np.abs)
        val = abs(details_flat_sorted[M])
        #print("threshold upper bound = ", val)

        # Calculating the percentage of threshold values
        sum_less = 0
        sum_more = 0
        for i in range(details.shape[0]):
            for j in range(details.shape[1]):
                if abs(details[i][j]) < val:
                    sum_less += 1
                elif abs(details[i][j]) >= val:
                    sum_more += 1

        #print("Total number of details = ", details.shape[1]*details.shape[0])
        #print("Number of elements to be compressed = ", sum_less)
        #print("Percentage of compression = ", sum_less/(details.shape[1]*details.shape[0])*100.)

        details[abs(details) < val] = float(0)

        # We put details and approximation back together
        index = 0
        for i in range(coeff_slices[0][0].stop, len(coeff_arr)):
            coeff_arr_threshold[i] = details[index]
            index += 1

        return coeff_arr_threshold, coeff_slices

    def from_dense_to_sparce(self, coeff_arr):
        """
        This method converts threshold coefficients to sparse matrix
        """
        coeff_arr_csr = csr_matrix(coeff_arr)
        return coeff_arr_csr

    def test_independent_frame_compression(self):
        """
        This method processes video frame-by-frame in test format.
        It takes a slice of video and independently processes each frame.
        Then it writes data to 4 files: "gs_coo_coeffs.txt", "gs_coeffs.txt", "frame.txt" and "gs_comp.mp4"
        "gs_csr_coeffs.txt" --- writes sparse coefficients and non-sparse slices to txt file
        "gs_coeffs.txt"     --- writes dense  coefficients and non-sparse slices to txt file
        "frame.txt"         --- writes initial, non-processed frame to the file
        "gs_comp.mp4"       --- reconstruction of decomposed and thresholded frame
                                for visual analysis of the algo
        Then it's interesting to compare sizes of three first files to observe the gain of compression.
        Returns average MSE of the (initial, compressed) videos.
        """
        mses = []  # Storage for MSE of each frame pair
        step = 5   # Number of frames to load to memory at once
        num_of_frames_to_process = int(self.stream.amount_of_frames)  # How many frames we want to compress

        # We use self.stream to write mp4 file
        self.stream.set_ofstream("write_mp4_grayscale", "./data/compressed/gs_comp.mp4")

        # And we create three extra stream objects to write down other video representations
        stream_coeff_sparse = VideoReader()
        stream_coeff_sparse.set_ofstream("write_txt", "./data/compressed/gs_csr_coeffs.txt")
        stream_coeff_sparse.write_header(num_of_frames_to_process)

        stream_coeff_dense  = VideoReader()
        stream_coeff_dense.set_ofstream("write_txt", "./data/compressed/gs_coeffs.txt")
        stream_coeff_dense.write_header(num_of_frames_to_process)

        stream_frame        = VideoReader()
        stream_frame.set_ofstream("write_txt", "./data/compressed/frame.txt")
        stream_frame.write_header(num_of_frames_to_process)

        index = 0
        for slice in range(0, num_of_frames_to_process, step):
            frames = self.stream.process_some_mp4frames(slice, slice + step, "return_grayscale")
            for frame in frames:
                print("Processing frame number ", index)
                # Compress and reconstruct frame

                coeffs1, slices1 = self.decompose_frame(frame)
                threshold_coeffs, slices = self.threshold_frame(coeffs1, slices1)
                composed_coeffs = self.from_dense_to_sparce(threshold_coeffs)
                reconstructed_frame = np.uint8(self.reconstruct_frame(threshold_coeffs, slices))
                mses.append(mse(frame, reconstructed_frame))
                # cv2.imshow("Initial frame", frame)
                # cv2.waitKey(0)
                # cv2.imshow("Reconstructed frame", reconstructed_frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # Write frame down
                self.stream.write_frame_mp4(reconstructed_frame)
                stream_coeff_sparse.write_sparse_coeffs_txt(composed_coeffs, slices, index)
                stream_coeff_dense.write_dense_coeffs_txt(threshold_coeffs, slices, index)
                stream_frame.write_frame_txt(frame, index)
                index += 1

        #print("sum(mses) / len(mses)", sum(mses) / len(mses))
        # Releasing streams
        self.stream.release_instream()
        self.stream.release_ofstream("write_mp4_grayscale")
        stream_coeff_sparse.release_ofstream("write_txt")
        stream_coeff_dense.release_ofstream("write_txt")
        stream_frame.release_ofstream("write_txt")
        return sum(mses) / len(mses)

    def test_decompression(self):
        """
        This method decodes video from the best-so-far compressed file,"gs_csr_coeffs.txt"
        which is of the way:
        # Total numbers of frame #
        # Sparse frame number #
        # Coefficients array shape: (,)
        # (i, j) value
        # Slices
        # Slice list
        # New frame ...
        Here we read the compressed file, decode it and writes to .mp4 format to be able to
        visually analyse it.
        """
        # We use self.stream to write mp4 file
        self.stream.set_ofstream("write_mp4_grayscale", "./data/compressed/gs_comp_from_csr.mp4")
        # stream_coeff_dense  = VideoReader()
        # stream_coeff_dense.set_ofstream("write_txt", "./data/compressed/gs_coeffs_decoded.txt")

        self.stream.set_instream("./data/compressed/gs_csr_coeffs.txt")

        total_frames = self.stream.read_header()
        #stream_coeff_dense.write_header(total_frames)
        #print("Total number of frames to decode = ", total_frames)
        for frame in range(total_frames):
            print("Decoding frame number ", frame)
            coeffs, slices = self.stream.read_sparse_txt_frame()
            reconstructed_frame = np.uint8(self.reconstruct_frame(coeffs, slices))
            #print("coeffs shape ", coeffs.shape)
            # cv2.imshow("Reconstructed frame", reconstructed_frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            self.stream.write_frame_mp4(reconstructed_frame)
            #stream_coeff_dense.write_dense_coeffs_txt(coeffs, slices, frame)
        self.stream.release_instream()
        self.stream.release_ofstream("write_mp4_grayscale")
        #stream_coeff_dense.release_ofstream("write_txt")
        self.stream.fin.close()


# Some useful debug #prints
        # #print("Initial frame", frame)
        # #print("Reconstructed frame", reconstructed_frame)
        # cv2.imshow("Initial frame", frame)
        # cv2.waitKey(0)
        # cv2.imshow("Reconstructed frame", reconstructed_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # #print("psnr = ", psnr(frame, reconstructed_frame))
        # #print("mse  = ", mse(frame, reconstructed_frame))

        # #print("Details arr length = ", coeff_slices[0][0].stop)
        # #print("Coef arr length = ", len(coeff_arr))
        # #print("Coef arr shape = ", coeff_arr.shape)
        # #print("Coef arr total = ", coeff_arr.shape[1] * coeff_arr.shape[0])