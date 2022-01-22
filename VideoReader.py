import cv2
import numpy as np
import matplotlib.pyplot as plt


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
            self.out = open(read_from, 'a')

    def set_instream_mp4(self, read_from):
        self.cap = cv2.VideoCapture(read_from)
        # self.cap = cv2.VideoCapture(0)

        # Check if stream opened successfully
        if self.cap.isOpened() == False:
            print("Error opening video stream or file")

        # Getting parameters of the video
        # self.frames_count, self.fps, self.width, self.height = self.cap.get(cv2.CAP_PROP_FRAME_COUNT), \
        #     self.cap.get(cv2.CAP_PROP_FPS), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.amount_of_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # self.fps = 23
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

    def set_ofstream(self, mode, write_to):
        if mode == "write_mp4":
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.out = cv2.VideoWriter(write_to, fourcc, self.fps, (self.frame_width, self.frame_height))
        elif "write_txt":
            self.out = open(write_to, 'a')

    def release_ofstream(self, mode):
        if mode == "write_mp4":
            self.out.release()
        elif "write_txt":
            self.out.close()

    def write_frame_mp4(self, frame):
        self.out.write(frame)

    def write_frame_txt(self, frame):
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        self.out.write('# Array shape: {0}\n'.format(frame.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in frame:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(self.out, data_slice, fmt='%-7.1f')

            # Writing out a break to indicate different slices...
            self.out.write('# New slice\n')

    def process_all_txtframes(self):
        print("To be done")

    def process_mp4frame(self):
        print("To be done")

    def process_some_mp4frames(self, read_from, start_frame_ind, end_frame_ind, mode="read", write_to=None):
        """
        This method reads some (start_frame_ind to end_frame_ind) frames from the mp4 stream
        """
        self.set_instream(read_from)
        print('mode = ', mode)
        if mode == "write_mp4" or mode == "write_txt":
            self.set_ofstream(mode, write_to)

        for frame_ind in range(start_frame_ind, end_frame_ind, 1):
            # check for valid frame number
            if frame_ind >= 0 & frame_ind <= self.amount_of_frames:
                # set frame position
                self.cap.set(int(self.amount_of_frames), frame_ind)
                ret, frame = self.cap.read()
                print("WE ret", ret)
                if ret:
                    if   mode == "write_mp4":
                        self.write_frame_mp4(frame)
                    elif mode == "write_txt":
                        self.write_frame_txt(frame)
                    elif mode == "show":
                        print("WE SHOW")
                        cv2.imshow("frame", frame)
                        cv2.waitKey(0)
                    elif mode == "process":
                        print("Processing frame")
                else:
                    break

        self.cap.release()
        if mode == "write_mp4" or mode == "write_txt":
            self.release_ofstream(mode)

    def process_all_mp4frames(self, read_from, mode="read", write_to=None):
        """
        This method reads all the frames from the mp4 stream
        """
        self.set_instream(read_from)
        if mode == "write_mp4" or mode == "write_txt":
            self.set_ofstream(mode, write_to)

        frame_index = 0
        while self.cap.isOpened():
            print("Frame index = ", frame_index)
            ret, frame = self.cap.read()
            if ret:
                if   mode == "write_mp4":
                    self.write_frame_mp4(frame)
                elif mode == "write_txt":
                    self.write_frame_txt(frame)
                elif mode == "show":
                    cv2.imshow("frame", frame)
                    cv2.waitKey(0)
                elif mode == "process":
                    print("Processing frame")
            else:
                break
            frame_index += 1

        self.cap.release()
        if mode == "write_mp4" or mode == "write_txt":
            self.release_ofstream(mode)