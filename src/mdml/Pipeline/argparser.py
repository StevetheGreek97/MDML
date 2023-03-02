import argparse as ap


parser = ap.ArgumentParser(description="Config file")


parser.add_argument("-c", "--config_file", 
                       dest = 'confg_file',
                       action="store",
                       required=True,
                       type=str,
                       help="gets the cofiguration file")