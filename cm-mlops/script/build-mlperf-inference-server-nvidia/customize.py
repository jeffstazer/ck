from cmind import utils
import os
import shutil

def preprocess(i):

    os_info = i['os_info']

    if os_info['platform'] == 'windows':
        return {'return':1, 'error': 'Windows is not supported in this script yet'}
    env = i['env']

    if '+LIBRARY_PATH' not in env:
        env['+LIBRARY_PATH'] = []
    env['+LIBRARY_PATH'].append(os.path.join(env['CM_TENSORRT_INSTALL_PATH'], "lib"))

    return {'return':0}

def postprocess(i):

    env = i['env']
    env['MLPERF_SCRATCH_PATH'] = os.getcwd()

    return {'return':0}
