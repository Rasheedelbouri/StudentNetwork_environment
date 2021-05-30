# -*- coding: utf-8 -*-
"""
Created on Sat May 29 20:27:12 2021

@author: rashe
"""


class Error(Exception):
    pass

class shapeError(Error):
    
    
    def __init__(self, message):
        self.message = message