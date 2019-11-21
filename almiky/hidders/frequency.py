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
        # insertion process
        for i in range(block_instace_8x8.max_num_blocks()):
            block8x8 = block_instace_8x8.get_block(i)
            block_transf8x8 = transform.direct(block8x8)
            vac = utils.vzig_zag_scan(block_transf8x8)
            for k in range(1,9):
                if l < len(bin_msg) - 1:
                    l += 1
                    if vac[k] < 0:
                        vac[k] = -utils.replace(
                            abs(round(vac[k])), bin_msg[l]
                        )
                    else:
                        vac[k] = utils.replace(round(vac[k]), bin_msg[la])
            block_transf8x8 = utils.mzig_zag_scan(vac)
            blocks_instance.set_block_image(
                transform.inverse(block_transf8x8), i
            )

        return watermarked_array

    def get_msg(self, watermarked_array):
        # Initial Values
        x = self.x0
        extracted_lsb = ""
        # Chaotic positions
        x = utils.pwlcm(x, self.p)
        pos = utils.random_list(x, self.p, list(range(1024)))[:self.n][256:]
        # Depth of the image
        if len(watermarked_array.shape) == 2:
            red_watermarked_array = watermarked_array
        else:
            # Red component
            red_watermarked_array = watermarked_array[:, :, 0]
        # Instance
        block_instace_32x32 = BlocksImage(red_watermarked_array, 32, 32)
        # Convert to integer sequence
        int_seq = list(map(int, self.sha512_key))
        # Watermark extraction process
        for i in range(block_instace_32x32.max_num_blocks()):
            block3x32 = block_instace_32x32.get_block(i)
            block_instace_1x1 = BlocksImage(block3x32, 1, 1)
            for j in range(256):
                extracted_lsb += utils.ext_lsb(
                    block_instace_1x1.get_block(pos[j])[0][0]
                )

        # Number of bits to determine the length of the secret message
        embd_cap = block_instace_32x32.max_num_blocks() * 256
        len_emb_cap = len(utils.base_change(embd_cap, 2))
        len_emb_bits = utils.bin2dec(extracted_lsb[:len_emb_cap])

        return utils.bin2char(extracted_lsb[len_emb_cap:][:len_emb_bits])

    def authenticate(self, watermarked_array):
        # Initial Values
        x = self.x0
        modified_blocks = []
        # Depth of the image
        if len(watermarked_array.shape) == 2:
            red_watermarked_array = watermarked_array
        else:
            # Red com# Binary message
        bin_msg = utils.char2bin(msg)
        # Chaotic positions
        x = utils.pwlcm(x, self.p)
        pos = utils.random_list(x, self.p, list(range(1024)))[:512]
        # Creating copy
        watermarked_array = np.copy(cover_array)
        # Depth of the image
        if len(watermarked_array.shape) == 2:
            red_watermarked_array = watermarked_array
        else:
            # Red component
            red_watermarked_array = watermarked_array[:, :, 0]
        # Instance
        block_instace_32x32 = BlocksImage(red_watermarked_array, 32, 32)
        # Checking the embedding capacity
        embd_cap = block_instace_32x32.max_num_blocks() * 256
        len_emb_cap = len(utils.base_change(embd_cap, 2))
        embd_cap -= len_emb_cap
        if len(bin_msg) > embd_cap:
            aux_mess = " exceeds the embedding capacity."
            raise ExceededCapacity
        bin_msg = utils.base_change(len(bin_msg), 2, len_emb_cap) + bin_msgponent
            red_watermarked_array = watermarked_array[:, :, 0]
        # Instance
        block_instace_32x32 = BlocksImage(red_watermarked_array, 32, 32)
        # Convert to integer sequence
        int_seq = list(map(int, self.sha512_key))
        # Calculating chaotic positions
        x = utils.pwlcm(x, self.p)
        pos = utils.random_list(x, self.p, list(range(1024)))[:self.n]
        pos_wm = pos[:256]
        # Watermark extraction process
        for i in range(block_instace_32x32.max_num_blocks()):
            block3x32 = block_instace_32x32.get_block(i)
            block_instace_1x1 = BlocksImage(block3x32, 1, 1)
            copy_1x1 = deepcopy(block_instace_1x1)
            for j in range(self.n):
                copy_1x1.set_block([[int_seq[j]]], pos[j])
            hash_seq = utils.sha256_to_image_bin(copy_1x1.get())
            extracted_lsb = ""
            for j in range(256):
                extracted_lsb += utils.ext_lsb(
                    block_instace_1x1.get_block(pos_wm[j])[0][0]
                )
            if extracted_lsb != hash_seq:
                modified_blocks.append(i)

        if modified_blocks:
            return False, modified_blocks

        return True,

    def process(self, cover_array, msg=None):
        # Initial value
        x = self.x0
        # Binary message
        bin_msg = utils.char2bin(msg)
        # Chaotic positions
        x = utils.pwlcm(x, self.p)
        pos = utils.random_list(x, self.p, list(range(1024)))[:512]
        # Creating copy
        watermarked_array = np.copy(cover_array)
        # Depth of the image
        if len(watermarked_array.shape) == 2:
            red_watermarked_array = watermarked_array
        else:
            # Red component
            red_watermarked_array = watermarked_array[:, :, 0]
        # Instance
        block_instace_32x32 = BlocksImage(red_watermarked_array, 32, 32)
        # Checking the embedding capacity
        embd_cap = block_instace_32x32.max_num_blocks() * 256
        len_emb_cap = len(utils.base_change(embd_cap, 2))
        embd_cap -= len_emb_cap
        if len(bin_msg) > embd_cap:
            aux_mess = " exceeds the embedding capacity."
            raise ExceededCapacity
        bin_msg = utils.base_change(len(bin_msg), 2, len_emb_cap) + bin_msg
        emb_size = utils.embedded_size(
            len(bin_msg),
            block_instace_32x32.max_num_blocks()
        )
        # Convert to integer sequence
        int_seq = list(map(int, self.sha512_key))
        # Watermark insertion process
        for i in range(block_instace_32x32.max_num_blocks()):
            block3x32 = block_instace_32x32.get_block(i)
            block_instace_1x1 = BlocksImage(block3x32, 1, 1)
            copy_1x1 = deepcopy(block_instace_1x1)
            for j in range(self.n):
                copy_1x1.set_block([[int_seq[j]]], pos[j])
            hash_seq = utils.sha256_to_image_bin(copy_1x1.get())
            diff = emb_size[i] - 256
            if diff == 256:
                hash_seq += bin_msg[i * diff:(i+1) * diff]
            elif diff != 0 and diff != 256:
                hash_seq += bin_msg[-diff:]
            for j in range(emb_size[i]):
                coef = utils.replace(
                    block_instace_1x1.get_block(pos[j])[0][0], hash_seq[j]
                )
                block_instace_1x1.set_block([[coef]], pos[j])

        return watermarked_array
