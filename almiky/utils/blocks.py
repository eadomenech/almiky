

class BlocksImage():
    '''
    Process an gray image in blocks

    To process an image in 8x8 block type:
    blocks = BlockImage(image, 8, 8)

    You can get an specific block by index in row major order:
    block = blocks[0]

    or set a block (blocks shape must match):
    block[0] = new_block

    You can modify individual block elements
    block[0][5] = 75

    This two las operation modify image elements.

    It is posible too iterate over the blocks:
    for block in blocks: block[2][7] = ...

    '''
    def __init__(self, image_plane, sblock_rows=8, sblock_cols=8):
        self.matrix = image_plane
        self.size_block_rows = sblock_rows
        self.size_block_cols = sblock_cols
        self.blocks_in_rows = self.matrix.shape[0] // self.size_block_rows
        self.blocks_in_cols = self.matrix.shape[1] // self.size_block_cols

    def get(self):
        return self.matrix

    def max_num_blocks(self):
        return self.blocks_in_rows * self.blocks_in_cols

    def image_size(self):
        return self.matrix.shape

    def get_coord(self, num_block):
        if num_block < self.max_num_blocks():
            L = []
            row_block = int(num_block / self.blocks_in_cols)
            col_block = num_block % self.blocks_in_cols
            L.append(row_block * self.size_block_rows)
            L.append((row_block + 1) * self.size_block_rows)
            L.append(col_block * self.size_block_cols)
            L.append((col_block + 1) * self.size_block_cols)
            return L

        raise IndexError("There is no such block")

    def __getitem__(self, num_block):
        '''
        Return the block specified in num_block argument.

        Arguments:
        num_block: block number

        Raises an IndexError if num_block is higher than
        blocks number of image or num_block is lesser than cero.
        '''
        pos = self.get_coord(num_block)
        return self.matrix[pos[0]:pos[1], pos[2]:pos[3]]

    def __setitem__(self, num_block, block):
        '''
        Replace a block.

        Arguments:
        num_block: block number
        block: replacement block

        Raises an IndexError if num_block is higher than
        blocks number of image or num_block is lesser than cero.
        '''
        pos = self.get_coord(num_block)
        self.matrix[pos[0]:pos[1], pos[2]:pos[3]] = block

    def __iter__(self):
        '''
        Implement iter(self)
        '''
        for i in range(self.max_num_blocks()):
            yield self[i]

    def __len__(self):
        return self.max_num_blocks()
