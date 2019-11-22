'''
Data hidding methods for digital images
'''

import os
import numpy as np
from copy import deepcopy
from almiky.utils import utils
from almiky.metrics import metrics
from almiky.utils.blocks_class import BlocksImage
from almiky.exceptions import ExceededCapacity
from almiky.moments.transform import Transform


class DataHiding():
    def process(self, data, msg):
        pass


class HidderEightFrequencyCoeficients():
    """
    Image is divided in 8x8 blocks and each block is transformed to
    frequency domain. One bit of message is inserted in least significant
    bit of 2-9 coficients of each block.
    """

    def __init__(self, ortho_matrix):
        '''
        obj.__init__(ortho_matrix) => None: Set quasi orthonormal
        matrix used frequency domain transformation.
        '''
        self.ortho_matrix = ortho_matrix

    def __verify_msg__(self, msg):
        pass

    def insert_msg(self, cover_array, msg=None):
        '''
        obj.insert(cover_array, msg) => (np.numpy): Return a watermarked
        work (stego work) with specified message.
        '''
        # Initial Values
        l = -1
        # Instance
        transform = Transform(ortho_matrix)
        # Binary message
        bin_msg = utils.char2bin(msg)
        # Creating copy
        watermarked_array = np.copy(cover_array)
        # Depth of the image
        if len(watermarked_array.shape) == 2:
            red_watermarked_array = watermarked_array
        else:
            # Red component
            red_watermarked_array = watermarked_array[:, :, 0]
        # Instance
        block_instace_8x8 = BlocksImage(red_watermarked_array)
        # Increasing the hash function
        # self.hashf = increase(self.hashf, block_instace_8x8.max_num_blocks())
        # Checking the embedding capacity
        embd_cap = block_instace_8x8.max_num_blocks() * 8
        len_emb_cap = len(utils.base_change(embd_cap, 2))
        embd_cap -= len_emb_cap
        if len(bin_msg) > embd_cap:
            aux_mess = " exceeds the embedding capacity."
            raise ExceededCapacity
        bin_msg = utils.base_change(len(bin_msg), 2, len_emb_cap) + bin_msg
        # Insertion process
        for i in range(block_instace_8x8.max_num_blocks()):
            block8x8 = block_instace_8x8.get_block(i)
            block_transf8x8 = transform.direct(block8x8)
            vac = utils.vzig_zag_scan(block_transf8x8)
            for k in range(1,9):
                if l < len(bin_msg) - 1:
                    l += 1
                    vac[k] = np.sing(vac[k]) * utils.replace(
                        abs(round(vac[k])), bin_msg[l]
                    )
                    # if vac[k] < 0:
                    #     vac[k] = -utils.replace(
                    #         abs(round(vac[k])), bin_msg[l]
                    #     )
                    # else:
                    #     vac[k] = utils.replace(round(vac[k]), bin_msg[la])
            block_transf8x8 = utils.mzig_zag_scan(vac)
            blocks_instance.set_block_image(
                transform.inverse(block_transf8x8), i
            )

        return watermarked_array


    def get_msg(self, stego_array, msg=None):
        '''
        obj.get(watermarked_array) => (np.numpy): Return the message.
        '''
        # Instance
        transform = Transform(ortho_matrix)
        # Binary message
        bin_msg = utils.char2bin(msg)
        # Depth of the image
        if len(watermarked_array.shape) == 2:
            red_watermarked_array = watermarked_array
        else:
            # Red component
            red_watermarked_array = watermarked_array[:, :, 0]
        # Instance
        block_instace_8x8 = BlocksImage(watermarked_array)
        # Extraction process
        for i in range(block_instace_8x8.max_num_blocks()):
            block8x8 = block_instace_8x8.get_block(i)
            block_transf8x8 = transform.direct(block8x8)
            vac = utils.vzig_zag_scan(block_transf8x8)
            for k in range(1,9):
                extracted_lsb += utils.ext_lsb(abs(round(vac[k])))

        # Number of bits to determine the length of the secret message
        embd_cap = block_instace_32x32.max_num_blocks() * 256
        len_emb_cap = len(utils.base_change(embd_cap, 2))
        len_emb_bits = utils.bin2dec(extracted_lsb[:len_emb_cap])

        return utils.bin2char(extracted_lsb[len_emb_cap:][:len_emb_bits])
