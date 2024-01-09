#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configparser
import os
import os.path

import pandas as pd
import yaml

cfg = configparser.ConfigParser()
cfg.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini'))
cfg = cfg._sections


class _EpriData(object):
    _wind_power = None
    _damping_power = None

    @staticmethod
    def wind_power() -> dict:
        if _EpriData._wind_power is None:
            filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    cfg['wind_power']['file'])
            _EpriData._wind_power = yaml.safe_load(open(filename, 'r'))
        return _EpriData._wind_power

    @staticmethod
    def damping_power() -> dict:
        if _EpriData._damping_power is None:
            filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    cfg['damping_power']['file'])
            _EpriData._damping_power = yaml.safe_load(open(filename, 'r'))
        return _EpriData._damping_power


class _FatigueData(object):
    _dat = None

    @staticmethod
    def qreg_input():
        if _FatigueData._dat is None:
            filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    cfg['fatigue_model']['file_qreg'])
            _FatigueData._dat = pd.read_csv(filename)
        return _FatigueData._dat
