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
        '-g', '--generate',
        help='Generate',
        action='store_true'
    )
    argparser.add_argument(
        '-n', '--convert',
        help='Convert',
        action='store_true'
    )
    args = argparser.parse_args()
    return args
