#!/usr/bin/env python3
# Every 60 seconds, launch a make command in the current directory

import os
import time

def main():
    while True:
        os.system('make')
        time.sleep(60)
        
if __name__ == '__main__':
    main()
