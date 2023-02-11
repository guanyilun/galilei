"""Console script for galilei."""

import fire


def help():
    print("galilei")
    print("=" * len("galilei"))
    print("the galilei project")


def main():
    fire.Fire({"help": help})


if __name__ == "__main__":
    main()  # pragma: no cover
