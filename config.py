import configparser

"""Obtain configurations of the experiments."""

config = configparser.ConfigParser()
config.read('./language-curiosity/config.conf')
# config.read('config.conf')
print('OPTIONS' in config)
# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
default_config = config[default]
