import pandas as pd

filepath = "spam.dat"

def main():
    print("Hello")
    print(filepath)
    file_content = pd.read_csv(filepath) # odzywam sie po naglowkach
    print(file_content['ACT_NOW']) 


if __name__ == "__main__":
    main()