from Preprocces import preprocess
from MachineLearning import ML
from Mapping import mapping
from  argparser import parser

def main():
    args = parser.parse_args()
    confg_file = args.confg_file
    preprocess(confg_file)
    ML()
    mapping()

if __name__ == "__main__":
    main()
