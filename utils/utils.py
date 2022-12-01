import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file'
    )
    argparser.add_argument(
        '-t', '--train',
        help='Train Model',
        action='store_true'
    )
    argparser.add_argument(
        '-e', '--Evaluate',
        help='Evaluate Model',
        action='store_true'
    )
    args = argparser.parse_args()
    return args
