'''
Data hidding methods for digital images
'''

import os
import numpy as np
from copy import deepcopy
from almiky.utils import utils
from almiky.utils.blocks import BlocksImage
from almiky.exceptions import ExceededCapacity


class HidderFrequency:
    """
    Image is divided in 8x8 blocks and each block is transformed to
    frequency domain.
    """

    def __init__(self, ortho_matrix):
        '''
        obj.__init__(ortho_matrix) => None: Set quasi orthonormal
        matrix for frequency domain transform.
        '''
        self.ortho_matrix = ortho_matrix

    def validate_capacity(self, bin_msg, embd_cap):
        len_emb_cap = len(utils.base_change(embd_cap, 2))
        embd_cap -= len_emb_cap
        if len(bin_msg) > embd_cap:
            raise ExceededCapacity
        #return utils.base_change(len(bin_msg), 2, len_emb_cap) + bin_msg


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

    def insert(self, cover_array, msg=None):
        '''
        obj.insert(cover_array, msg) => (np.numpy): Return a watermarked
        work (stego work) with specified message.
        '''
        # Initial Values
        l = -1
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
        if len(bin_msg) > embd_cap:
            raise ExceededCapacity

        for i in range(block_instace_8x8.max_num_blocks()):
            block8x8 = block_instace_8x8[i]
            block_transf8x8 = self.ortho_matrix.direct(block8x8)
            vac = utils.vzig_zag_scan(block_transf8x8)
            for k in range(1,9):
                if l < len(bin_msg) - 1:
                    l += 1
                    vac[k] = np.sign(vac[k]) * utils.replace(
                        abs(round(vac[k])), bin_msg[l]
                    )
            block_transf8x8 = utils.mzig_zag_scan(vac)
            block_instace_8x8[i] = self.ortho_matrix.inverse(block_transf8x8)

        return watermarked_array


    def extract(self, ws_array, msg=None):
        '''
        obj.get(watermarked_array) => (np.numpy): Return the message.
        '''
        extracted_lsb = ''
        # Depth of the image
        if len(ws_array.shape) == 2:
            red_ws_array = ws_array
        else:
            # Red component
            red_ws_array = ws_array[:, :, 0]
        # Instance
        block_instace_8x8 = BlocksImage(red_ws_array)
        # Extraction process
        for i in range(block_instace_8x8.max_num_blocks()):
            block8x8 = block_instace_8x8[i]
            block_transf8x8 = self.ortho_matrix.direct(block8x8)
            vac = utils.vzig_zag_scan(block_transf8x8)
            for k in range(1,9):
                extracted_lsb += utils.ext_lsb(abs(round(vac[k])))
        return utils.bin2char(extracted_lsb)


class HidderFrequencyLeastSignificantBit(HidderFrequency):
    def __insert__(self, bit, block, index):
        block_transf8x8 = self.ortho_matrix.direct(block)
        coeficients = block_transf8x8.reshape(-1)
        coeficients[index] = (
            np.sign(coeficients[index]) *
            utils.replace(abs(round(coeficients[index])), bit)
        )
        return block_transf8x8

    def __extract__(self, block, index):
        block = self.ortho_matrix.direct(block)
        coeficient = block.reshape(-1)[index]
        return utils.ext_lsb(coeficient)

    def insert(self, cover_array, msg, coeficient_index):
        # Binary data
        bin_msg = utils.char2bin(msg)
        # Cover copy and block generation
        watermarked_array = np.copy(cover_array)
        block_instance_8x8 = BlocksImage(watermarked_array)

        # Checking the embedding capacity
        embd_cap = block_instance_8x8.max_num_blocks()
        self.validate_capacity(bin_msg, embd_cap)

        # insertion process
        for index, bit in enumerate(bin_msg):
            block8x8 = block_instance_8x8.get_block(index)
            block_transf8x8 = self.__insert__(bit, block8x8, coeficient_index)
            block_instance_8x8.set_block(
                self.ortho_matrix.inverse(block_transf8x8), index)

        return watermarked_array

    def extract(self, watermarked_array, coeficient_index):
        msg = ''
        block_manager = BlocksImage(watermarked_array)
        for block in range(block_manager.max_num_blocks()):
            msg += self.__extract__(block, coeficient_index)

        return utils.bin2char(msg)


class BlockHider:
    """
    Hides data in transform domain using a desired embedder.
    Cover work is divided in block and bits of message are
    hiden in coeficients of trasnformed blocks.
    """

    def __init__(self, embedder, transform):
        """
        Set embedder an transform used by hider

        Arguments:
        embedder -- It allows to embed message bits in the coeficients.
        transform -- It allows to move from the domain of space
        to the trasform domain and vice versa
        """
        self.embedder = embedder
        self.transform = transform

    def hide(self, cover, msg):
        """
        Hide msg in cover work and return watermarked/stego work

        Arguments:
        cover -- cover work
        msg -- binary message coded in an str
        """
        ws_work = np.copy(cover)

        return cover
