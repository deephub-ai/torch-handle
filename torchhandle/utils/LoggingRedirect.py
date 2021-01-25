import sys

class LoggingRedirect(object):

    def __init__(self, filename):

        self.terminal = sys.stdout

        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def start(filename):
    """Start transcript, appending print output to given filename"""
    sys.stdout = LoggingRedirect(filename)

def stop():
    """Stop transcript and return print functionality to normal"""
    if hasattr(sys.stdout,"logfile"):
        sys.stdout.logfile.close()
    if hasattr(sys.stdout, "terminal"):
        sys.stdout = sys.stdout.terminal