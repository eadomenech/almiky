'''
Multiple objective funtions
'''


class WeightedAggretation:
    '''
    Conventional weighted aggregation. Usefull
    when weights remains fixed during the run of
    the algorithm.

    Create an instance from weighted objective functions:
    wf = WeightedAggregation(f, g)

    then evaluate functions with custom arguments.
    '''
    def __init__(self, functions):
        '''
        Initialize self. See help(type(self)) for accurate signature.

        Arguments:
        data -- (callable): objective functions.

        Raise A ValueError if functions argument is empty.
        '''
        if not functions:
            raise ValueError("function list empty")

        self.functions = functions

    def __call__(self, *args, **kwargs):
        '''
        Return weighted combination of all objective functions.
        '''
        values = (f(*args, **kwargs) for f in self.functions)
        return sum(values)
