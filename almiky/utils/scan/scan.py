'''Methods for scanning'''


from .maps import ROW_MAJOR_8x8


class ScanMapper:
    '''
    Provide an interface for scan a NxN data using a map

    You can use default row major 8x8 map:
    scanning = ScanMapper(data)

    or use a custom map:
    scanning = ScanMapper(data, map=ZIGZAG)

    Then is posible get an element by index:
    amplitude = scanning[5]

    or modify an element
    scanning[2] = 0.75

    A IndexError is raised if index is out of range.

    Also is posible iterate over elements:
        for coefficient in scanning:
            ...

    '''

    def __init__(self, data, map):
        '''
        Initialize self. See help(type(self)) for accurate signature.

        Arguments:
        data -- data to scan
        map -- map used for scanning
        '''
        self.data = data
        self.map = map
        self.length = self.data.shape[1]

    def _get_indexes(self, pos):
        '''
        Return an coefficient

        Arguments:
        pos -- map index
        '''
        x = int(pos / self.length)
        y = pos % self.length

        if x >= self.length:
            raise IndexError('block indexes out of range')

        return x, y

    def get_pos(self, index):
        '''
        Return coefficient position in block

        Arguments:
        index -- map index
        '''
        try:
            pos = self.map[index]
        except IndexError as e:
            raise IndexError('scan index out of range') from e

        return pos

    def __iter__(self):
        for i in self.map:
            x, y = self._get_indexes(i)
            yield self.data[x, y]

    def __getitem__(self, index):
        '''
        Return a coeficient

        Arguments:
        index -- coefficient index
        '''
        pos = self.get_pos(index)
        x, y = self._get_indexes(pos)

        return self.data[x, y]

    def __setitem__(self, index, value):
        '''
        Return a coeficient

        Arguments:
        index -- coefficient index
        '''
        pos = self.get_pos(index)
        x, y = self._get_indexes(pos)

        self.data[x, y] = value


class ScanMapping:
    '''
    Allows to obtain an ScanMapper given a NxN data
    and a scan map.

    The mapping is built given a scan map
    scan = ScanMapping(map=ZIGZAG)

    or using default row major map:
    scan = ScanMapping()

    The ScanMapper instance is obtained calling scan with
    data as argument.
    scanning = scan(data)

    See help(ScanMapper) for more details
    '''
    def __init__(self, map=ROW_MAJOR_8x8):
        '''
        Initialize self. See help(type(self)) for accurate signature.

        Arguments:
        map -- map used for scanning
        '''
        self.map = map

    def __call__(self, data):
        '''
        Return an scan mapper

        Arguments:
        data -- data to scan: (NxN) numpy array
        '''
        return ScanMapper(data, self.map)
