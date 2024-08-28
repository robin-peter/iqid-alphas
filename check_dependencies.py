try:
    print('Your version (Suggested version)')
    import numpy as np
    print('numpy: {} (1.20.2)'.format(np.__version__))
    import cv2
    print('cv2: {} (4.0.1)'.format(cv2.__version__))
    import scipy
    print('skimage: {} (0.18.1)'.format(scipy.__version__))
    import skimage
    print('scipy: {} (1.6.2)'.format(skimage.__version__))
    import pystackreg
    print('pystackreg: {} (0.2.5)'.format(pystackreg.__version__))
    import matplotlib
    print('matplotlib: {} (3.3.4)'.format(matplotlib.__version__))

    # iQID packages
    from iqid import process_object, align, dpk
    print('All dependencies successfully located. Double-check versions if anything breaks.')
except Exception as e:
    print('Missing dependency: ' + repr(e))