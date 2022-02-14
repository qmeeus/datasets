

def test_dataset():
    from datasets import Dataset
    grabo = Dataset("config/grabo.2.json")
    pp10 = grabo("pp10")
    print(len(pp10))
    print(pp10.input_dim, pp10.output_dim)


def test_grabo():
    from datasets import Grabo
    grabo = Grabo()
    pp10 = grabo("pp10")
    print(len(pp10))
    print(pp10.input_dim, pp10.output_dim)


if __name__ == "__main__":
    test_grabo()