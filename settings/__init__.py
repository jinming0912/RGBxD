import logging

from settings.base_setting import BaseSetting
from settings.deeplab_v3_setting import DeeplabV3Setting
from settings.deeplab_v3_plus_setting import DeeplabV3PlusSetting
from settings.u_net_setting import UNetSetting


def load_setting(path):
    if path == 'DeeplabV3':
        return DeeplabV3Setting(None)
    elif path == 'UNet':
        return UNetSetting(None)
    elif path == 'DeeplabV3Plus':
        return DeeplabV3PlusSetting(None)
    else:
        logging.error("Unknown setting name.")
        return None