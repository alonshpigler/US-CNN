import numpy as np
from scipy.signal import hilbert
import h5py
from enum import Enum
import matplotlib.pyplot as plt


class DispType(Enum):
    PEAK_TO_PEAK = 1
    MAX = 2
    MIN = 3
    MIN_MAX = 4
    ENVELOP_PEAK = 5
    ENVELOP_PEAK_DB = 6
    ENVELOP_TIME_PEAK = 7
    MAX_TO_MAX = 8
    NUM_DETECTIONS = 9
    FIRST_TWO = 10


class HdfData:
    def __init__(self, hdf_file_name):
        self.a_scan_mat = None

        with h5py.File(hdf_file_name, 'r') as file:

            self.phi_arr = file['Phi Array'][:]
            self.radius_arr = file['Radius Array'][:]
            self.z_arr = file['Z Array'][:]
            a_scans_dataset = file['A-Scans'][:, :]
            self.a_scan_mat = np.float64(a_scans_dataset)
            if a_scans_dataset.dtype is np.dtype('uint8'):
                self.a_scan_mat /= np.power(2., 8)
            elif a_scans_dataset.dtype is np.dtype('uint16'):
                self.a_scan_mat /= np.power(2., 16)
            else:
                raise NotImplementedError
            self.a_scan_mat -= np.mean(self.a_scan_mat)

            self.sample_rate = file['A-Scans'].attrs['Sample Rate'] / 1e6
            self.is_3D = np.bool(file.attrs['Is 3D'])
            self.x_arr = self.radius_arr * np.cos(self.phi_arr)
            self.y_arr = self.radius_arr * np.sin(self.phi_arr)
            self.reso = self.get_scan_reso()

            if self.is_3D:
                raise NotImplementedError
                # curvDistArr =  self.phi_arr * self.radius_arr
                # midDist = (np.max(curvDistArr) - np.min(curvDistArr))/2
                # curvDistArr -= midDist
            else:
                self.j_arr = np.uint64(np.floor((self.x_arr - np.min(self.x_arr)) / self.reso))
                self.i_arr = np.uint64(np.floor((self.y_arr - np.min(self.y_arr)) / self.reso))

            self.num_col = np.uint64(np.max(self.j_arr) + 1)
            self.num_row = np.uint64(np.max(self.i_arr) + 1)
            self.wave_indx_mat = np.zeros((self.num_row, self.num_col), dtype='uint64')
            for indx, i_indx in enumerate(self.i_arr):
                j_indx = self.j_arr[indx]
                self.wave_indx_mat[i_indx, j_indx] = indx

        print('end')

    def get_s_pos_arr(self):
        raise NotImplementedError

    def get_scan_reso(self):
        if self.is_3D:
            if abs(self.phi_arr[0] - self.phi_arr[1]) > 1e-16:
                arc0 = self.phi_arr[0] * self.radius_arr[0]
                arc1 = self.phi_arr[1] * self.radius_arr[1]
                return abs(arc0 - arc1)
            else:
                # This is for the rotor which scans first the Z axis and then phi
                return abs(self.z_arr[0] - self.z_arr[1])
        else:
            delta_x = self.x_arr[1:] - self.x_arr[:-1]
            delta_y = self.y_arr[1:] - self.y_arr[:-1]
            dist_arr = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2.0))
            first_change_indx = (dist_arr > 1e-6).nonzero()[0][0]
            dist = dist_arr[first_change_indx]
            return dist

    def arrange_a_scans_in_mat(self):
        raise NotImplementedError

    @staticmethod
    def get_disp_val(vals_arr, disp_type):
        if disp_type is DispType.PEAK_TO_PEAK:
            val = np.max(vals_arr) - np.min(vals_arr)
        elif disp_type is DispType.MAX:
            val = np.max(vals_arr)
        elif disp_type is DispType.MIN:
            val = np.min(vals_arr)
        elif disp_type is DispType.MIN_MAX:
            min_val = np.abs(np.min(vals_arr))
            max_val = np.abs(np.max(vals_arr))
            val = max(min_val, max_val)
        elif disp_type is DispType.ENVELOP_PEAK:
            envelop = np.abs(hilbert(vals_arr))
            val = np.max(envelop)
        elif disp_type is DispType.ENVELOP_PEAK_DB:
            envelop = np.abs(hilbert(vals_arr))
            envelop = 20 * np.log10(envelop)
            val = np.max(envelop)
        elif disp_type is DispType.ENVELOP_TIME_PEAK:
            #envelop = np.abs(hilbert(vals_arr))
            val = np.argmax(vals_arr)
        elif disp_type is DispType.MAX_TO_MAX:
            sorted_vals_arr = np.argsort(np.abs(vals_arr))
            val = np.abs(sorted_vals_arr[-1] - sorted_vals_arr[-2])
        elif disp_type is DispType.NUM_DETECTIONS:
            val = np.count_nonzero(vals_arr)
        elif disp_type is DispType.FIRST_TWO:
            non_zero_inds = np.nonzero(vals_arr)
            if len(non_zero_inds[0]) > 1:
                val = non_zero_inds[0][1] - non_zero_inds[0][0]
            else:
                val = 0
        else:
            val = 0

        return val

    def get_c_scan(self, val_type=DispType.PEAK_TO_PEAK, n0=0, n1=None):
        (num_wave, wave_len) = self.a_scan_mat.shape
        if n1 is None:
            n1 = wave_len - 1

        if n1 not in range(0, wave_len):
            raise IndexError

        c_scan = np.zeros_like(self.wave_indx_mat, dtype='float64')
        for indx, i_indx in enumerate(self.i_arr):
            j_indx = self.j_arr[indx]
            a_scan = self.a_scan_mat[indx][n0:n1]
            c_scan[i_indx, j_indx] = HdfData.get_disp_val(a_scan, val_type)

        return c_scan

    def get_a_scan(self, i_indx, j_indx=None):

        if j_indx is None:
            return self.a_scan_mat[i_indx, :]
        else:
            index = self.wave_indx_mat[i_indx, j_indx]
            return self.a_scan_mat[index, :]

    def get_ascan_ind(self, i_indx, j_indx):
        return self.wave_indx_mat[i_indx, j_indx]

if __name__ == '__main__':
    file_name = 'C:/phantom1-5MHz_Focus_N01_Glue_Interface.hdf'
    hdf_data = HdfData(file_name)
    cur_c_scan = hdf_data.get_c_scan(DispType.ENVELOP_TIME_PEAK, 0, 180)
    cur_a_scan = hdf_data.get_a_scan(100, 100)
    plt.figure('c-scan')
    plt.imshow(cur_c_scan)
    plt.figure('a-scan')
    plt.plot(cur_a_scan)
    plt.show()

