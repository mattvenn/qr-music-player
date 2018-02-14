import subprocess
import os

def add(album):
    proc = subprocess.Popen(['/usr/bin/mpc','current','-q','-f','%album%'], stdout=subprocess.PIPE)
    current_album = proc.stdout.read().strip()
    if album == current_album:
        print("already playing %s" % album)
    else:
        print("adding %s" % album)
        os.system('mpc clear; mpc search any %s | mpc add; mpc play' % album)

