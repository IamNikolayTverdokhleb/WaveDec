import cv2
import numpy as np
import re
from scipy.sparse import csr_matrix
import ast

class VideoReader:
    def __init__(self):
        """
        This class is a wrapper around opencv and numpy read-write routines
        to read and write frames in mp4 or txt formats
        """
        # TODO add more supported formats

    def set_instream(self, read_from):
        """
        This method sets up all needed parameters to read frames from the \
        file depending on the type of the file
        """
        if read_from.find("mp4") > -1:
            print("Reading from .mp4")
            self.set_instream_mp4(read_from)
        else:
            print("Reading from .txt")
            self.fin = open(read_from, 'r')

    def set_instream_mp4(self, read_from):
        """
        This method sets up all needed parameters to read video from .mp4
        """

        self.cap = cv2.VideoCapture(read_from)
        # self.cap = cv2.VideoCapture(0) # To stream from webcam

        # Check if stream opened successfully
        if self.cap.isOpened() == False:
            print("Error opening video stream or file")

        # Getting parameters of the video
        self.amount_of_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

    def get_stream_params(self):
        """
        This method returns all parameters of the stream
        """
        return self.amount_of_frames, self.fps, self.frame_width, self.frame_height

    def set_ofstream(self, mode, write_to):
        """
        This method sets up ofstream
        """
        self.write_to = write_to
        if mode == "write_mp4_grayscale":
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.out = cv2.VideoWriter(write_to, fourcc, self.fps, (self.frame_width, self.frame_height), 0)
        elif mode == "write_mp4":
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.out = cv2.VideoWriter(write_to, fourcc, self.fps, (self.frame_width, self.frame_height))
        elif "write_txt":
            self.out = open(write_to, 'w')

    def release_instream(self):
        """
        This method kills instream objects
        """
        self.cap.release()
        #self.fin.close()

    def release_ofstream(self, mode):
        """
        This method kills ofstream objects wrt to the mode
        """
        if mode == "write_mp4" or mode == "write_mp4_grayscale":
            self.out.release()
        elif "write_txt":
            self.out.close()

    def write_frame_mp4(self, frame):
        """
        This method writes a frame into mp4 file
        """
        self.out.write(frame)

    def write_frame_txt(self, frame, index=None):
        """
        This method writes a frame into txt file
        """
        self.out.write('# Frame number {0}\n'.format(index))
        self.out.write('# Array shape: {0}\n'.format(frame.shape))

        for data_slice in frame:
            np.savetxt(self.out, data_slice, fmt='%-7.1f')
        self.out.write('# End of frame\n')

    def write_dense_coeffs_txt(self, composed_coeffs, slices, index=None):
        """
        This method dense coefficients and slices to the txt file
        """
        self.out.write('# Dense frame number {0}\n'.format(index))
        self.out.write('# Coefficients array shape: {0}\n'.format(composed_coeffs.shape))
        for data_slice in composed_coeffs:
            np.savetxt(self.out, data_slice, fmt='%-7.1f')
        self.out.write('\n# End of coefficients array\n')

        self.out.write('# Slices\n')
        for item in slices:
            self.out.write("{}\n".format(item))
        self.out.write('# End of slices\n')
        self.out.write('# End of frame\n')

    def write_sparse_coeffs_txt(self, composed_coeffs, slices, index=None):
        """
        This method sparse coefficients and slices to the txt file
        """
        self.out.write('# Sparse frame number {0}\n'.format(index))
        self.out.write('# Coefficients array shape: {0}\n'.format(composed_coeffs.shape))
        composed_coeffs.maxprint = composed_coeffs.count_nonzero()
        self.out.write(str(composed_coeffs))
        self.out.write('\n# End of coefficients array\n')

        self.out.write('# Slices\n')
        for item in slices:
            self.out.write("{}\n".format(item))
        self.out.write('# End of slices\n')
        self.out.write('# End of frame\n')

    def process_all_txtframes(self):
        print("To be done")

    def process_mp4frame(self):
        print("To be done")

    def process_some_mp4frames(self, start_frame_ind, end_frame_ind, mode="read", write_to=None):
        """
        This method reads some (start_frame_ind to end_frame_ind) frames from the mp4 stream
        """
        if mode == "write_mp4" or mode == "write_mp4_grayscale" or mode == "write_txt":
            self.set_ofstream(mode, write_to)
        elif mode == "return_grayscale":
            storage = []

        for frame_ind in range(start_frame_ind, end_frame_ind, 1):
            # check for valid frame number
            if frame_ind >= 0 & frame_ind <= self.amount_of_frames:
                # set frame position
                self.cap.set(int(self.amount_of_frames), frame_ind)
                ret, frame = self.cap.read()
                if ret:
                    if mode == "write_mp4" or mode == "write_mp4_grayscale":
                        self.write_frame_mp4(frame)
                    elif mode == "write_txt":
                        self.write_frame_txt(frame)
                    elif mode == "show":
                        cv2.imshow("frame", frame)
                        cv2.waitKey(0)
                    elif mode == "return_grayscale":
                        storage.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                        # storage.append(frame)
                    elif mode == "process":
                        print("Processing frame")
                else:
                    break
        if mode == "return_grayscale":
            return storage

    def process_all_mp4frames(self, mode="read", write_to=None):
        """
        This method reads all the frames from the mp4 stream
        """
        if mode == "write_mp4" or mode == "write_mp4_grayscale" or mode == "write_txt":
            self.set_ofstream(mode, write_to)
        elif mode == "return_grayscale":
            storage = []

        frame_index = 0
        while self.cap.isOpened():
            print("Frame index = ", frame_index)
            ret, frame = self.cap.read()
            if ret:
                if mode == "write_mp4" or mode == "write_mp4_grayscale":
                    self.write_frame_mp4(frame)
                elif mode == "write_txt":
                    self.write_frame_txt(frame)
                elif mode == "show":
                    cv2.imshow("frame", frame)
                    cv2.waitKey(0)
                elif mode == "return_grayscale":
                    storage.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                elif mode == "process":
                    print("Processing frame")
            else:
                break
            frame_index += 1

        if mode == "return_grayscale":
            return storage

    def write_header(self, number):
        self.out.write('# Total number of frames {0}\n'.format(number))

    def read_header(self):
        line = self.fin.readline()
        return int(re.findall(r'\d+', line)[0])

    def read_sparse_txt_frame(self):
        line = self.fin.readline()
        data = []
        row = []
        col = []
        slices = []
        end_state = False
        while line:
            line = self.fin.readline()
            # Read sparse coefficients
            if line.find("# Coefficients array shape:") != -1:
                # print("Read sparse coefficients")
                shape1 = int(re.findall(r'\d+', line)[0])
                shape2 = int(re.findall(r'\d+', line)[1])
                bool = True
                while bool:
                    line = self.fin.readline()
                    if line.find("# End of coefficients array") != -1:
                        bool = False
                    else:
                        str = re.findall(r'\d+', line)
                        row.append(int(str[0]))
                        col.append(int(str[1]))
                        data.append(float(str[2] + '.' + str[3]))
                        if line.find("-") != -1:
                            data[-1] = -data[-1]

            # Read slices
            elif line.find("# Slices") != -1:
                # print("Read slices")
                bool_slices = True
                while bool_slices:
                    line = self.fin.readline()
                    if line.find("# End of slices") != -1:
                        bool_slices = False
                        end_state = True
                    else:
                        slices.append(eval(line.strip("\n")))
            # Done with the frame
            elif end_state:
                break
        # print('Retrieved matrix shape: {0} {1}\n'.format(shape1, shape2))
        # creating sparse matrix
        sparseMatrix = csr_matrix((data, (row, col)),
                                  shape=(shape1, shape2), dtype=np.float).toarray()
        return sparseMatrix, slices
