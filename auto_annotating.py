import sys
from ultralytics.data.annotator import auto_annotate


def annotate_it(det_model: str, sam_model: str, input_dir: str, output_dir: str, min_conf: str):
    auto_annotate(det_model=det_model,
                  sam_model=sam_model,
                  output_dir=output_dir,
                  data=input_dir,
                  conf=float(min_conf))


def main():
    argv = sys.argv
    annotate_it(*argv[1:])
    pass


if __name__ == '__main__':
    main()
