class WeightedFunction:
    '''
    Allow to get a weighted version of an arbitrary function:
    wf = WeightedFunction(f, 0.5)

    wf(5) => 0.5 * f(5)
    '''
    def __init__(self, function, weight):
        '''
        Initialize self. See help(type(self)) for accurate signature.

        Arguments:
        function -- (callable): a function
        weight -- weight
        '''
        self.function = function
        self.weight = weight

    def __call__(self, *args, **kwargs):
        '''
        Return weighted function evaluation.
        '''
        return self.weight * self.function(*args, **kwargs)
