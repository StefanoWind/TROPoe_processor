# -*- coding: utf-8 -*-
"""
General utilities
"""
import numpy as np

def create_logger(filename,level='info'):
    import logging
    # Create a logger
    logger = logging.getLogger()
    
    if level=='info':
        logger.setLevel(logging.INFO)
    if level=='debug':
        logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.hasHandlers():
        logger.addHandler(handler)
    
    return logger, handler

def close_logger(logger, handler):
    # Remove handler
    logger.removeHandler(handler)
    # Close handler
    handler.close()

def dt64_to_num(dt64):
    '''
    Converts Unix timestamp into numpy.datetime64
    '''
    tnum=(dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return tnum

def datenum(string,format="%Y-%m-%d %H:%M:%S.%f"):
    '''
    Turns string date into unix timestamp
    '''
    from datetime import datetime
    num=(datetime.strptime(string, format)-datetime(1970, 1, 1)).total_seconds()
    return num

def datestr(num,format="%Y-%m-%d %H:%M:%S.%f"):
    '''
    Turns Unix timestamp into string
    '''
    from datetime import datetime
    string=datetime.utcfromtimestamp(num).strftime(format)
    return string
    
def round(x,resolution):
    '''
    Round with resolution
    '''
    return np.round(x/resolution)*resolution

def floor(x,resolution):
    '''
    Floor with resolution
    '''
    return np.floor(x/resolution)*resolution

def hstack(a,b):
    '''
    Stack vectors horizontally
    '''
    if len(np.shape(b))==1:
        b=np.reshape(b,(len(b),1))
    if len(a)>0:
        ab=np.hstack((a,b))
    else:
        ab=b
    return ab


def vstack(a,b):
    '''
    Stack vectors vertically
    '''
    if len(a)>0:
        ab=np.vstack((a,b))
    else:
        ab=b
    return ab 

def sind(x):
    '''
    Sine in degrees
    '''
    return np.sin(x/180*np.pi)
    