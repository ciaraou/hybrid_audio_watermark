class Window():
    """
    Window class. if win_func is set ignore win_type. One must be set. 

    Atributes:
        win_len : int 
            length of window
        win_func : function 
            window function default None
        win_type : string
            alternative windowing method default None
    """
    def __init__(self, win_len, win_func=None, win_type=None):
        """
        initializes Window object

        Parameters
        ----------
        win_len : int
            length of window
        win_func : function
            windowing function default None
        win_type : strign
            alternative windowing method default None (ignored by class if win_func is set)
        """
        self.N = win_len 
        self.win_func = win_func
        self.win_type = win_type

    def apply_window(self, N, x): # TODO fix this with self.N change
        """
        applies window function to x

        Parameters
        ----------
        x : array_like
            sound input

        Exception
        ---------
        Exception
            windowing function or type needed
        
        """
        if self.win_func != None:
            window = self.win_func(2*N) # todo precompute this, restructure objects around x
            return x * window
        elif self.win_type != None:
            raise Exception("windowing type not implemented yet")
        else:
            raise Exception("windowing function or type needed")
