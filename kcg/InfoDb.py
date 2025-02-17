import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
from kcg.Kernel import *

class CacheManagerST :
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CacheManagerST, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass