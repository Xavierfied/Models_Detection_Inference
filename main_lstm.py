from runners.lift_runner import run
from utils.args import get_lift_args


def main() -> None:
    config = get_lift_args()
    source = 0 if config.source == '0' else config.source

    try:
        run(source, config)
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
