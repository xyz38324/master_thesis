from  traintest01 import main

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-max_episode', type=int, default=200)
    parser.add_argument('-evaluate', type=bool, default=True)
    parser.add_argument('-max_step', type=int, default=5000)

    #
    parser.add_argument('-alpha', type=float, default=1.5e-4)
    parser.add_argument('-beta', type=float, default=1.5e-3)
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-gamma', type=float, default=0.98)
    parser.add_argument('-tau', type=float, default=1e-2)
    parser.add_argument('-buffer_size', type=int, default=1000000)
    parser.add_argument('-save_interval', type=int, default=50)
    parser.add_argument('-learn_interval', type=int, default=100)
    args = parser.parse_args()
    main(args)