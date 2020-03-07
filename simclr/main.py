import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    args = parser.parse_args()
    print(args.run_name)

if __name__ == '__main__':
    main()