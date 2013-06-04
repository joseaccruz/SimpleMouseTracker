import argparse
import sys
import time

class Argument(object):
    def __init__( self, name, atype, cmd_name, cmd_option=None, default="", desc="", minimum=0, maximum=99 ):
        self.name = name
        self.desc = desc
        self.atype = atype
        self.default = default
        self.cmd_name = cmd_name
        self.cmd_option = cmd_option
        self.minimum = minimum
        self.maximum = maximum

        self._widget = None

    def add_argument(self, parser):
        if self.cmd_option is None:
            parser.add_argument( self.cmd_name, help=self.desc )
        else:
            if self.atype == "bool":
                parser.add_argument( self.cmd_option, self.cmd_name, help=self.desc, action="store_%s" %(self.default and "true" or "false") )
            else:
                atypes = { "dir":str, "file":str, "int":int, "str":str }
                parser.add_argument( self.cmd_option, self.cmd_name, type=atypes[self.atype], help=self.desc, default=self.default )


class Command(object):
    def __init__( self, name ):
        self.name = name
        self.args = []

        self._script = None

    def add_arg( self, arg ):
        self.args.append( arg )

    def get_parser( self ):
        parser = argparse.ArgumentParser()

        for arg in self.args:
            arg.add_argument( parser )

        return parser

class InOutAbstract(object):
    def start_progress( self, total ):
        self._total = total
        self._start = time.time()

    def show_progress( self, count ):
        if count == 0:
            count += 1

        self._deltat = time.time() - self._start
        stept = self._deltat / float(count)
        self._finalt = float(self._total-count) * stept


        self._deltats = time.strftime( "%H:%M:%S", time.gmtime(self._deltat) )
        self._finalts = time.strftime( "%H:%M:%S", time.gmtime(self._finalt) )

class InOutTerminal(InOutAbstract):
    def __init__( self ):
        pass

    def show( self, msg ):
        sys.stdout.write( "%s" %msg )
        sys.stdout.flush()

    def error( self, msg, fatal=True ):
        if fatal:
            sys.stderr.write( "\n!! ERROR: %s\n" %msg )
            #raise Exception()
            quit()
        else:
            sys.stderr.write( "\n!! WARNING: %s\n" %msg )

    def show_progress( self, count ):
        super(InOutTerminal, self).show_progress( count )

        back = "\b" * 75
        self.show( "%sFrame: %5d/%5d (%5.2f%%), Ellap: %s, Expec: %s" %(back, count, self._total, (float(count) / float(self._total)) * 100.0, self._deltats, self._finalts) )
